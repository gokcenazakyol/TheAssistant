from tkinter import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import random
import os
from playsound import playsound
import webbrowser as wb
import requests
from bs4 import BeautifulSoup
import time
from tkinter import messagebox

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
model = AutoModelForCausalLM.from_pretrained('gokcenazakyol/GokcenazGPT-small-v1')

# reset entry
def reset_entry():
    entry.delete(0, END)

# Define a function to return the Input data
def get_data():
    response = pointer()
    label.config(text=response, font=('Helvetica 13'), wraplength=500)
    response = response.replace("The Assistant: ", "")
    label.pack(pady=30)
    label.update_idletasks()

    label.after(100, lambda: speak_assistant(response))
    reset_entry()

def web_scraping(qs):
    global flag2
    global loading

    URL = 'https://www.google.com/search?q=' + qs
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')

    links = soup.findAll("a")
    all_links = []
    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            all_links.append((link.get('href').split("?q=")[1].split("&sa=U")[0]))

    flag = False
    for link in all_links:
        if 'https://en.wikipedia.org/wiki/' in link:
            wiki = link
            flag = True
            break

    div0 = soup.find_all('div', class_="kvKEAb")
    div1 = soup.find_all("div", class_="Ap5OSd")
    div2 = soup.find_all("div", class_="nGphre")
    div3 = soup.find_all("div", class_="BNeawe iBp4i AP7Wnd")

    if len(div0) != 0:
        return div0[0].text
    elif len(div1) != 0:
        return div1[0].text + "\n" + div1[0].find_next_sibling("div").text
    elif len(div2) != 0:
        return div2[0].find_next("span").text + "\n" + div2[0].find_next("div", class_="kCrYT").text
    elif len(div3) != 0:
        return div3[1].text
    elif flag == True:
        page2 = requests.get(wiki)
        soup = BeautifulSoup(page2.text, 'html.parser')
        title = soup.select("#firstHeading")[0].text

        paragraphs = soup.select("p")
        for para in paragraphs:
            if bool(para.text.strip()):
                return title + "\n" + para.text
    return ""


def command(input):
    words = input.split()
    for word in words:
        # LinkedIn
        if "linkedin" == word or "LinkedIn" == word:
            wb.open_new("https://www.linkedin.com/feed/")
            response = "Opening LinkedIn"
            return response

        # Google
        if "google" == word or "Google" == word:
            wb.open_new("https://www.google.com.tr/?hl=tr")
            response = "Opening Google"
            return response

        # Youtube
        if "youtube" == word or "YouTube" == word:
            wb.open_new("https://www.youtube.com")
            response = "Opening YouTube"
            return response

    return ""


def speak_assistant(response):
    if response != "":
        tts = gTTS(response, lang='en')
        rand = random.randint(1, 1000)
        file = 'audio' + str(rand) + '.mp3'
        tts.save(file)
        playsound(file)
        os.remove(file)


def pointer():
    input = entry.get()
    response = command(input)
    if response != "":
        return "The Assistant: " + response
    response2 = web_scraping(input)
    if response2 != "":
        return "The Assistant: " + response2
    for step in range(1):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


    return "The Assistant: " + response

def clock():
    hour = time.strftime("%H")
    minute = time.strftime("%M")
    second = time.strftime("%S")

    clock_label.config(text=hour+":"+minute+":"+second)
    clock_label.after(1000, clock)


root = Tk()
root.title("The Assistant")
root.geometry('700x250')

bg = PhotoImage(file='wallpaper2.png')
bg_label = Label(root, image=bg)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create Clock widget
clock_label = Label(root, text="", font=("Helvetica", 28), fg="grey")
clock_label.pack(pady=20)

# Create an Entry Widget
entry = Entry(root, width=42)
entry.place(relx=.5, rely=.7, anchor=CENTER)

label = Label(root, text="", font=('Helvetica 13'))
# Create a Button to get the input data
Button(root, text="Send", command=get_data).place(relx=.82, rely=.7, anchor=CENTER)
# Button(root, text="Talk", command=get_data).place(relx=.9, rely=.5, anchor=CENTER)


# I will update this version with a checklist
"""def toggle_menu():
    if var.get():
        print("Toggle menu is on")
        # Code to execute when the toggle menu is turned on
    else:
        print("Toggle menu is off")
        # Code to execute when the toggle menu is turned off

var = BooleanVar()

toggle_button = Checkbutton(root, text="Toggle Menu", variable=var, command=toggle_menu)
toggle_button.pack()"""


clock()
root.mainloop()
