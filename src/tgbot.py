import telebot
from textextractor.image2text import image_to_text
from summarization.summarizer import summarize

bot = telebot.TeleBot('6870572126:AAEQWCgzcSjYesarrTLlav0t97rzKIVvI7U')

@bot.message_handler(content_types=['text', 'photo'])
def get_text_messages(message):
    if message.text == "Hello":
        bot.send_message(message.from_user.id, "Hello dear user")
    elif message.caption == "/summarize":
        way = download_photo(message)
        text = image_to_text(way).replace('\n', '')
        summarization = summarize(text)
        bot.send_message(message.from_user.id, summarization)
    else:
        bot.send_message(message.from_user.id, "I don't understand you")


def download_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    way = f'src/photo/img{message.photo[len(message.photo) - 1].file_id}.jpg'
    with open(f'{way}', 'wb') as new_file:
        new_file.write(downloaded_file)
    return way

bot.polling(none_stop=True)