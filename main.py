from telegram.ext import Updater, CommandHandler

def start(update, context):
    update.message.reply_text('Привет! Я — СкальперБот. Готов к бою!')

def help_command(update, context):
    update.message.reply_text('Пиши /start чтобы начать. Остальное пока в разработке.')

def main():
    updater = Updater("7720722235:AAHcVuYNlQkNzXKTcmJDMnkGQHgE95L02OE", use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()