from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.filters.command import CommandObject
from aiogram.types import Message

from rag_tg_project.rag.rag_chain import get_user_question_and_return_answer 


router = Router()



@router.message(CommandStart())
async def command_start(message: Message):
    await message.reply(f"The bot is running.")
    return


@router.message()
async def ai_handler(message: Message):
    user_question = message.text.strip()

    answer = get_user_question_and_return_answer(user_question=user_question)

    await message.answer(answer)

