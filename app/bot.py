import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

from rag_tg_project.app.handlers import router


load_dotenv()
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

dp = Dispatcher()


async def main():
    bot = Bot(token=TG_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)

    await dp.start_polling(bot)



logging.basicConfig(level=logging.INFO)
asyncio.run(main())