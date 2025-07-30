from loguru import logger
from monarchmoney import MonarchMoney

from mmac.amazon_connector.amazon_order_connector import AmazonOrderConnector
from mmac.amazon_connector.types import AmazonOrderData
from mmac.captcha_solver.llm_captcha_solver import LLMCaptchaSolver
from mmac.config.types import AmazonAccount, Config
from mmac.fsm.state_machine_implementation import OrderScraperFSM
from mmac.monarch_connector.monarch import MonarchConnector


class MonarchMoneyAmazonConnectorCLI:
    def __init__(self, config: Config):
        self._config = config

        self._captcha_solver = None

        self._fsm = OrderScraperFSM()
        self._orders: AmazonOrderData = None  # type: ignore

        if self._config.llm.enable_llm_captcha_solver:
            self._captcha_solver = LLMCaptchaSolver(
                openai_api_key=self._config.llm.api_key,
                model_name=self._config.llm.llm_model_name,
                base_url=self._config.llm.base_url,
                project=self._config.llm.project,
                organization=self._config.llm.organization,
            )

    async def _get_monarch_money(self, use_saved_session: bool = True) -> MonarchMoney:
        self._mm = MonarchMoney()

        logger.info("No Monarch Money session found. Logging in.")
        logger.debug(f"Logging in with email: {self._config.monarch_account.email}")
        await self._mm.login(
            email=self._config.monarch_account.email,
            password=self._config.monarch_account.password,
            mfa_secret_key=self._config.monarch_account.mfa_secret_key,
            use_saved_session=use_saved_session,
        )

        return self._mm

    async def _get_amazon_connector(
        self, account: AmazonAccount
    ) -> AmazonOrderConnector:
        connector = AmazonOrderConnector(
            username=account.email,
            password=account.password,
            mfa_secret_key=account.mfa_secret_key,
            headless=self._config.headless,
            pause_between_navigation=self._config.debug.pause_between_navigation,
            captcha_solver=self._captcha_solver,
            searchFilter=self._config.amazon_filter,
        )

        logger.info("Retrieving Amazon orders.")

        if self._fsm.current_state.id == self._fsm.all_orders_scraped.id:
            self._fsm.reset()

        self._fsm.send("stay_on_login", amazon=connector)

        if self._fsm.orders is None:
            raise Exception(
                f"Failed to retrieve orders for Amazon account: {account.email}"
            )
        else:
            self._orders = self._fsm.orders

        return connector

    async def _annotate_single_account(
        self,
        amazon_connector: AmazonOrderConnector,
        monarch_connector: MonarchConnector,
    ):
        username = amazon_connector._username

        logger.info(f"Annotating transactions found in Amazon Account: {username}")

        logger.debug(
            f"Found {len(self._orders.orders)} orders for Amazon account: {username}"
        )

        logger.debug(
            f"Matching transactions to Amazon orders for Amazon account: {username}"
        )
        transactions = await monarch_connector.get_transactions_need_review()
        transaction_mapping = await monarch_connector.match_transactions_to_amazon(
            self._orders, transactions=transactions
        )

        logger.debug(f"Adding notes to Amazon orders for Amazon account: {username}")
        await monarch_connector.add_notes_to_amazon_orders(matches=transaction_mapping)

    async def annotate_transactions(self):
        logger.info(
            f"Annotating transactions across {len(self._config.amazon_accounts)} Amazon accounts."
        )

        monarch_connector = MonarchConnector(
            monarch_money=await self._get_monarch_money(), config=self._config
        )

        try:
            await monarch_connector.validate_session()
        except Exception:
            monarch_connector = MonarchConnector(
                monarch_money=await self._get_monarch_money(use_saved_session=False),
                config=self._config,
            )
            await monarch_connector.validate_session()

        for account in self._config.amazon_accounts:
            amzn_conn = await self._get_amazon_connector(account=account)
            await self._annotate_single_account(
                amazon_connector=amzn_conn, monarch_connector=monarch_connector
            )
