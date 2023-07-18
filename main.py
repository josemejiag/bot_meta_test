import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import requests
import openai
import time
import json
import os
import shutil
import re
import io
import ast
from io import BytesIO
from pydub import AudioSegment
import warnings
import traceback
import http.client
import aiohttp
from aiogoogle import Aiogoogle
from aiohttp import ClientSession
import base64
from PIL import Image,PngImagePlugin
import ffmpeg
#import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import random
import webuiapi
import asyncio
import aiofiles
import aiosmtplib
from asyncio import Queue
from asyncio import Lock
from serpapi import GoogleSearch
import gspread
from pathlib import Path
from collections import deque
import glob
############ LIBRERIAS MAIL ##################
import smtplib
import zipfile
from email.message import EmailMessage
from email.mime.application import MIMEApplication
############ LIBRERIAS SHEETS ################
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
from google.oauth2 import service_account
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, tool, initialize_agent,AgentType
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMMathChain, SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.tools import BaseTool
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import gspread_asyncio

print('imported libraries')

#######################################################################################################################################################################################
#######################################################################################################################################################################################
#########################  TEXT FUNCTIONS ########################################

async def create_text_followup_chatGPT(last5_msg,last5_sender,prompt,number,sender_id,whatsapp_key,openai_key,language,last_msg_type,last_msg,img_caption,img_process='SD'): 

    headers_openai = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }    
    openai.api_key = openai_key
    
    messages_chat = [{"role": "system", "content": "You are the most advanced and powerful conversational AI in the world. You will use your whole AGI potential to answer the User. You are helpful, creative and super intelligent, as well as your answers. Prefer to give short/medium length answers, unless complex answer is required or asked. Never evade a question. If the User speaks in Spanish answer in Spanish. Your are trained up to 2021 events. Today is "+ str(date.today())},]

    for i in range(len(last5_msg)):
        if last5_sender[i]=='AI':
            messages_chat.append({"role": "assistant", "content": str(last5_msg[i][:])})
        else:
            messages_chat.append({"role": "user", "content": str(last5_msg[i][:])})

    data = {
        "model": CHATGPT_MODEL,
        "messages": messages_chat,
        "max_tokens": 3000,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions",
                                headers=headers_openai, json=data) as response:
            result = await response.json()
    text = str(result['choices'][0]['message']['content']) 

    headers = {
        'Authorization': 'Bearer ' + whatsapp_key,
        #'Content-Type': 'application/json'
    }
    json = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": text
                }
            }    
    async with aiohttp.ClientSession() as session:
        async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                            headers=headers, json=json) as response:
            response = await response.json()

    return response

async def create_text_followup_GPT4(last5_msg,last5_sender,prompt,number,sender_id,whatsapp_key,openai_key,language,last_msg_type,last_msg,img_caption,img_process='SD'): 

    headers_openai = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }    
    openai.api_key = openai_key
    
    messages_chat = [{"role": "system", "content": "You are the most advanced and powerful conversational AI in the world. You will use your whole AGI potential to answer the User. You are helpful, creative and super intelligent, as well as your answers. Prefer to give short/medium length answers, unless complex answer is required or asked. Never evade a question. If the User speaks in Spanish answer in Spanish. Your are trained up to 2021 events. Today is "+ str(date.today())},]

    for i in range(len(last5_msg)):
        if last5_sender[i]=='AI':
            messages_chat.append({"role": "assistant", "content": str(last5_msg[i][:])})
        else:
            messages_chat.append({"role": "user", "content": str(last5_msg[i][:])})

    data = {
        "model": CHATGPT_MODEL,
        "messages": messages_chat,
        "max_tokens": 3000,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions",
                                headers=headers_openai, json=data) as response:
            result = await response.json()
    text = str(result['choices'][0]['message']['content']) 

    headers = {
        'Authorization': 'Bearer ' + whatsapp_key,
        #'Content-Type': 'application/json'
    }
    json = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": text
                }
            }   
    
    async with aiohttp.ClientSession() as session:
        async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                            headers=headers, json=json) as response:
            response = await response.json()

    return response

#######################################################################################################################################################################################
#######################################################################################################################################################################################
######################### WHATSAPP SEND API FUNCTIONS ########################################

async def send_text(prompt,number,sender_id,whatsapp_key):
    headers = {
        'Authorization': 'Bearer ' + whatsapp_key,
        #'Content-Type': 'application/json'
    }

    json = {
          "messaging_product": "whatsapp",
          "recipient_type": "individual",
          "to": number,
          "type": "text",
          "text": {
            "preview_url": False,
            "body": prompt
            }
        }
    async with aiohttp.ClientSession() as session:
        async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                            headers=headers, json=json) as response:
            response = await response.json()

async def send_image(link,number,sender_id,whatsapp_key):
    headers = {
        'Authorization': 'Bearer ' + whatsapp_key,
        #'Content-Type': 'application/json'
    }

    json = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": number,
        "type": "image",
        "image": {
        "link" : link
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                            headers=headers, json=json) as response:
            response = await response.json()

    
async def send_return_menu_button(conv,number,text,language):
    headers = {
        'Authorization': 'Bearer ' + whatsapp_key,
        #'Content-Type': 'application/json'
    }
    if language=='/es':
        title="Retornar al menu ↩️"
    else:
        title="Return to menu ↩️"

    json = {
              "messaging_product": "whatsapp",
              "recipient_type": "individual",
              "to": number,
              "type": "interactive",
              "interactive": {
                "type": "button",
                "body": {
                  "text": text
                },
                "action": {
                  "buttons": [
                    {
                      "type": "reply",
                      "reply": {
                        "id": "menu",
                        "title": title
                      }
                    }                   
                  ]
                }
              }
            }
    async with aiohttp.ClientSession() as session:
        async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                            headers=headers, json=json) as response:
            response = await response.json()
    conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),'return button','RETURN MENU',time.time()]
    conv.to_csv(root+'Data/Messages/'+str(number)+'.csv',index=False)  


async def send_plantilla_menu(conv,number,language):
     
    headers = {
        'Authorization': 'Bearer ' + whatsapp_key,
        #'Content-Type': 'application/json'
    }
    text = "1️⃣ *chatGPT3*\n2️⃣ *chatGPT4*"
    json = {
              "messaging_product": "whatsapp",
              "recipient_type": "individual",
              "to": str(number),
              "type": "interactive",
              "interactive": {
                "type": "list",
                "header": {
                  "type": "text",
                  "text": "MENÚ"
                },
                "body": {
                  "text": text
                },
                "footer": {
                  "text": "Ferreycorp"
                },
                "action": {
                  "button": "Elegir modo",
                  "sections": [ 
                    {
                      "title": "GPT models",
                      "rows": [
                        {
                          "id": "13",
                          "title": "ChatGPT-3",
                          "description": " "
                        },
                        {
                          "id": "14",
                          "title": "ChatGPT-4",
                          "description": " "
                        },                                   
                      ]
                    },

                  ]
                }
              }
            }

    async with aiohttp.ClientSession() as session:
        async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                            headers=headers, json=json) as response:
            response = await response.json()

    conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),'options menu','PLANTILLA0',time.time()]
    conv.to_csv(root+'Data/Messages/'+str(number)+'.csv',index=False)          
    

async def chatGPT_notification_v2(language,conv,number,mode,whatsapp_key,tool_name):

        prompt = "¡Hola! En que te puedo ayudar?"

        button1 = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": int(number),
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {
                "text": prompt
                },
                "action": {
                "buttons": [
                    {
                    "type": "reply",
                    "reply": {
                        "id": "menu",
                        "title": "Regresar al menú ↩️"
                    }
                    },
                ]
                }
            }
            }
        headers = {
            'Authorization': 'Bearer ' + whatsapp_key,
            #'Content-Type': 'application/json'
        }
        async with aiohttp.ClientSession() as session:
            async with session.post('https://graph.facebook.com/v15.0/'+ sender_id +'/messages',
                                headers=headers, json=button1) as response:
                response = await response.json()
        conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),prompt,mode,time.time()]
        conv.to_csv(root+'Data/Messages/'+str(number)+'.csv',index=False)   

#######################################################################################################################################################################################
#######################################################################################################################################################################################
#########################  AI RESPONSE FUNCTION ########################################
        
async def ai_response(last5_msg,last5_sender,last_msg,number,mode,conv,language,cat,img_caption,last_msg_type):

    global error_count
    global whatsapp_key
    global sender_id
    global OPENAI_API_KEY_1
    global nothing
    global model

    if last_msg == 'menu' or last_msg == 'Menu' or last_msg == 'Menú' or last_msg == 'men' or last_msg == 'exit' or last_msg == 'Exit' or last_msg == 'salir' or last_msg == 'Salir' or last_msg == 'return' or last_msg == 'Return': #USER SELECTED MODE CREATE TEXT              
        mode = '/aichat2'
        await send_plantilla_menu(conv,number,language)
    
    else :            
        if mode == '/chatGPT':
            prompt = last_msg
            text = await create_text_followup_chatGPT(last5_msg=last5_msg,last5_sender=last5_sender,prompt=prompt,number=number,sender_id=sender_id,whatsapp_key=whatsapp_key,openai_key=OPENAI_API_KEY_2,language=language,last_msg_type=last_msg_type,last_msg=last_msg,img_caption=img_caption)

            conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),text,mode,time.time()]
            conv.to_csv(root+'Data/Messages/'+str(number)+'.csv', encoding='utf-8',index=False) 

        elif mode == '/chatGPT4':
            prompt = last_msg
            text = await create_text_followup_GPT4(last5_msg=last5_msg,last5_sender=last5_sender,prompt=prompt,number=number,sender_id=sender_id,whatsapp_key=whatsapp_key,openai_key=OPENAI_API_KEY_2,language=language,last_msg_type=last_msg_type,last_msg=last_msg,img_caption=img_caption)

            conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),text,mode,time.time()]
            conv.to_csv(root+'Data/Messages/'+str(number)+'.csv', encoding='utf-8',index=False) 

        elif mode == '/aichat2':
            
            prompt = last_msg
            if last_msg.isdigit() or last_msg.isdecimal() or last_msg.isnumeric():
                if last_msg=='13':
                    mode = '/chatGPT'
                    cat = 'cat13'
                    await chatGPT_notification_v2(language,conv,number,mode,whatsapp_key,tool_name='GPT3')

                elif last_msg=='14': #GPT4
                    mode = '/chatGPT4'
                    cat = 'cat14'
                    await chatGPT_notification_v2(language,conv,number,mode,whatsapp_key,tool_name='GPT4')
                else:
                    cat = ''
                    await send_plantilla_menu(conv,number,language)
            else:
                cat = ''
                await send_plantilla_menu(conv,number,language)                
    
    return mode,language,cat,credits

#######################################################################################################################################################################################
#######################################################################################################################################################################################
#########################  TRANSCRIBE AUDIO ########################################
async def transcribe_audio_v2(audio_id, contact,language):
    global error_count
    global whatsapp_key
    global sender_id
    global OPENAI_API_KEY_1
    global nothing
    global model

    try:
        headers = {
            'Authorization': 'Bearer ' + whatsapp_key,
            'Content-Type': 'application/json'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get('https://graph.facebook.com/v15.0/'+audio_id+'/', headers=headers) as response:
                response = await response.json()
        audio_id_url = response['url']
        async with aiohttp.ClientSession() as session:
            async with session.get(audio_id_url, headers=headers) as response:
                if response.status == 200:
                    audio_content = await response.read()
                    with open('Data/Messages/audios/transcription_audio_' + contact + '.mp3', 'wb') as f:
                        f.write(audio_content)
                else:
                    print('audio transcribe : ', response.status)

        input_audio = ffmpeg.input('Data/Messages/audios/transcription_audio_'+contact+'.mp3',loglevel="quiet")
        lighter_audio = ffmpeg.output(input_audio, 'Data/Messages/audios/output_transcription_audio_'+contact+'.mp3', y='-y')
        ffmpeg.run(lighter_audio)
        
        audio_file = open('Data/Messages/audios/output_transcription_audio_'+contact+'.mp3', "rb")
        url = "https://api.openai.com/v1/audio/transcriptions"  
        result = openai.Audio.transcribe("whisper-1", audio_file)
        text = result['text']

        return text
    except Exception as e:
        print(traceback.format_exc())
        contacts = pd.read_csv(root+'Data/ID/'+str(contact)+'.csv')
        i = contacts.loc[contacts['Phone'] == int(contact)].index[0]
        conv = pd.read_csv(root+'Data/Messages/'+ contact +'.csv')        
        conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),'Error ','ERROR',time.time()]
        conv.to_csv(root+'Data/Messages/'+contact+'.csv',index=False)
        contacts.loc[i,['answer_state']] = 'idle'
        contacts.to_csv(root+'Data/ID/'+str(contact)+'.csv',index=False)

        return 'Transcription Error'

#######################################################################################################################################################################################
#######################################################################################################################################################################################
#########################  AUX FUNCTIONS ########################################   


def validate_string(string, a, b):
    if string.isdigit() and len(string) == 8 and a <= int(string) <= b:
        return True
    return False

def extract_numbers(string):
    numbers = re.findall(r'\d+', string)
    if numbers:
        return True, "".join(numbers)
    else:
        return False, ""
    
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None
    
def try_last_x_msg(conv,x_msg):
    last5_msg = []
    last5_sender = []
    conv = conv[(conv['type']!='audio')]
    for i in range(x_msg):
        try:
            if i<x_msg-1:
                last5_msg.append(conv['text_body'][-x_msg+i:-x_msg+i+1].values.tolist()[0])
                last5_sender.append(conv['from'][-x_msg+i:-x_msg+i+1].values.tolist()[0])
            else :
                last5_msg.append(conv['text_body'][-1:].values.tolist()[0])
                last5_sender.append(conv['from'][-1:].values.tolist()[0])                
        except:
            nothing=0
 
    return last5_msg,last5_sender
    
 
async def add_unknown(cel):

    contacts = pd.read_csv(root+'Data/ID/contacts.csv')
                                  #Phone	headers	plantilla	mode	language	plantilla0	plantilla_balde => credits, user, status, plan, answer_state
    contacts.loc[len(contacts)] = [cel,cel,True,True,'/aichat2','/es',False,False,' ',9999999,'idle']
    
    new_conv = pd.DataFrame(columns=['id','disp_phone_n','phone_n_id','contacts_name' ,'contacts_wa_id' ,'from','wa_msg_id','timestamp','text_body','type','timestamp2'])
    #new_conv.loc[len(new_conv)] = ['id','disp_phone_n','phone_n_id','AI' ,'contacts_wa_id' ,'AI','wa_msg_id','timestamp','text_body','CREATE MSG']
    await send_plantilla_menu(new_conv,cel,'/es')
    new_conv.to_csv(root+'Data/Messages/'+str(cel)+'.csv',index=False)
    contacts.to_csv(root+'Data/ID/'+str(cel)+'.csv',index=False)

def get_worksheet_row_count(sheet_id, sheet_name, creds):
    service = discovery.build('sheets', 'v4', credentials=creds)
    result = service.spreadsheets().get(spreadsheetId=sheet_id, ranges=sheet_name, fields="sheets(properties(gridProperties(rowCount)))").execute()
    return result['sheets'][0]['properties']['gridProperties']['rowCount']

# Set up the Google Sheets and Drive API clients
def create_google_api_clients():
    credentials = service_account.Credentials.from_service_account_file(
        'GCP_coe.json',
        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])

    sheets_api = build('sheets', 'v4', credentials=credentials)
    drive_api = build('drive', 'v3', credentials=credentials)

    return sheets_api, drive_api

sheets_api, drive_api = create_google_api_clients()

def get_last_modification_time(spreadsheet_id):
    try:
        file_metadata = drive_api.files().get(fileId=spreadsheet_id, fields='modifiedTime').execute()
        return file_metadata.get('modifiedTime')
    
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

# Helper function to convert a timestamp string to a datetime object
def parse_timestamp(timestamp_str):
    return date.fromisoformat(timestamp_str.replace('Z', '+00:00'))

print('loaded functions')
###########################################################################  DEFINE ENV VARS #################################################################


# API KEYS AND ENV VARS
#root='gs://botify-cloud-storage-00/'
root=''
whatsapp_syst_user_token1 = 'Exxxxxxxxxxxxxxxxxxxxxx' #Sebastian 1
whatsapp_syst_user_token2 = 'Exxxxxxxxxxxxxxxxxxxxxx'
whatsapp_key = 'Exxxxxxxxxxxxxxxxxxxxxx'
meta_admin_token = 'Exxxxxxxxxxxxxxxxxxxxxx'
whatsapp_key = whatsapp_syst_user_token2
#whatsapp_key = meta_admin_token
number = '519000000'
sender_id = '1050200000000'
with open('openai.txt', 'r') as file:
    data = file.read()
OPENAI_API_KEY_1 = str(data)
OPENAI_API_KEY_2 = OPENAI_API_KEY_1
time_duration = 1
last_ai_message = ''
last_human_message = ''
last_msg = ''
img_caption = ''
batch_n = 100
batch_pandas=''
nothing=False
error_count = 0
contacts = pd.read_csv(root+'Data/ID/contacts.csv')
scope = "https://spreadsheets.google.com/feeds"
credentials = ServiceAccountCredentials.from_json_keyfile_name('GCP_coe.json', scope)
previous_row_count = -1
row_count = 0
GPT4_MODEL = 'gpt-4'
CHATGPT_MODEL = 'gpt-3.5-turbo'
incoming_messages = asyncio.Queue()
incoming_messages_semaphore = asyncio.Semaphore(0)
worker_status = [None] * 5
processed_messages = deque(maxlen=100)
processed_message_ids = deque(maxlen=25)
new_messages_event = asyncio.Event()
first_message_processing = False
first_batch = True 

#########################  AWAIT PER USER INTERACTION FUNCTION ########################################
async def interact_async(contact_number):

                global error_count
                global whatsapp_key
                global sender_id
                global OPENAI_API_KEY_1
                global nothing
                global model
                global img_caption

                contacts = pd.read_csv(root+'Data/ID/'+str(contact_number)+'.csv')
                i = contacts.loc[contacts['Phone'] == int(contact_number)].index[0]
                conv = pd.read_csv(root+'Data/Messages/'+ contact_number +'.csv')

                language = contacts['language'][i]
                inactivity_timer = 60*60*24*7
                if contacts['answer_state'][i] != 'idle':
                    print('still processing')
                    await send_text("_*Por favor espera la respuesta antes de mandar otro mensaje*_",str(contacts['Phone'][i]),sender_id,whatsapp_key)
                    return
                else:
                    print('start processing')
                    contacts.loc[i,['answer_state']] = 'processing'
                    contacts.to_csv(root+'Data/ID/'+str(contact_number)+'.csv',index=False)

                if time.time() - (inactivity_timer) > int(conv['timestamp2'][-2:].values.tolist()[0]) and (conv['type'][-1:].values.tolist()[0]!='EXIT' or conv['type'][-2:].values.tolist()[0]!='PLANTILLA0'): #timer de sesion
                    print('paso1')
                    await send_text('_La conversación se reinicia despues de ' + str(np.round(inactivity_timer/60/60/24,2)) + ' dias de inactividad_' ,str(contacts['Phone'][i]),sender_id,whatsapp_key)
                    contacts.loc[i,['mode']] = '/aichat2'
                    contacts.loc[i,['plantilla0']] = False
                    contacts.loc[i,['answer_state']] = 'idle'
                    contacts.to_csv(root+'Data/ID/'+str(contact_number)+'.csv',index=False)  
                    conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),'EXIT','EXIT',time.time()]
                    conv.to_csv(root+'Data/Messages/'+contact_number+'.csv',index=False)

                elif contacts['plantilla0'][i]==True  : #si escribe algo en plantilla menu
                    print('paso2')
                    await send_plantilla_menu(conv,contact_number,language)
                    contacts.loc[i,['plantilla0']] = False
                    contacts.loc[i,['answer_state']] = 'idle'
                    contacts.to_csv(root+'Data/ID/'+str(contact_number)+'.csv',index=False)  

                elif conv['type'][-1:].values.tolist()[0]=='PLANTILLA0' or conv['type'][-1:].values.tolist()[0]=='INVALID ID': # hacer nada
                    print('paso4')
                    nothing=False
                    contacts.loc[i,['answer_state']] = 'idle'
                    contacts.to_csv(root+'Data/ID/'+str(contact_number)+'.csv',index=False)  

                else:               
                    print('paso5')
                    contacts.loc[i,['plantilla0']] = False
                    if contacts['headers'][i] == True :

                        try:
                            last_msg  = conv['text_body'][-1:].values.tolist()[0]    
                            last_sender = conv['from'][-1:].values.tolist()[0]
                            actual_mode = contacts['mode'][i]   

                            if actual_mode == '/chatGPT':
                                last5_msg,last5_sender = try_last_x_msg(conv,10) 
                            else:
                                last5_msg,last5_sender = try_last_x_msg(conv,5)   

                            
                            cat = contacts['category'][i]

                            if last_sender != 'AI':
                                #transcribe audio if needed
                                last_msg_type  = conv['type'][-1:].values.tolist()[0] 
                                if last_msg_type=='audio':
                                    audio_id = conv['text_body'][-1:].values.tolist()[0]
                                    last_msg = await transcribe_audio_v2(audio_id=audio_id,contact= last_sender,language = contacts['language'][i])
                                    if last_msg == 'Transcription Error':
                                        return

                                    conv.loc[len(conv)] = ['01','nan','nan','Human' ,'nan' ,contacts['Phone'][i],'nan',time.time(),last_msg,'transcription',time.time()]
                                    conv.to_csv(root+'Data/Messages/'+contact_number+'.csv', encoding='utf-8',index=False)
                                    last5_msg[-1] = last_msg

                                mode,language,cat,new_credits = await ai_response(last5_msg,last5_sender,last_msg,last_sender,contacts['mode'][i],conv,contacts['language'][i],cat,img_caption,last_msg_type)
                                contacts.loc[i,['category']] = cat
                                contacts.loc[i,['mode']] = mode
                                contacts.loc[i,['language']] = language
                                contacts.loc[i,['credits']] = new_credits
                                contacts.loc[i,['answer_state']] = 'idle'
                                contacts.to_csv(root+'Data/ID/'+str(contact_number)+'.csv',index=False)
                                print('answered - ',contacts['Phone'][i]) 


                        except Exception as e:
                            print('ai_response error : ')
                            print(traceback.format_exc())
                            conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),'Error','ERROR',time.time()]
                            conv.to_csv(root+'Data/Messages/'+contact_number+'.csv',index=False)
                            contacts.loc[i,['answer_state']] = 'idle'
                            contacts.to_csv(root+'Data/ID/'+str(contact_number)+'.csv',index=False)

################################################################################################################################################################
##############################################################################  MAIN LOOP ####################################################################################
         
async def process_batch():
    async with aiohttp.ClientSession() as session:
        worksheet = gspread.authorize(credentials).open_by_key('1su6WIP6L3qtOR4zQNZoPr95SCgFrtWQru5F4mf1zRe8').worksheet('Hoja 1')
        data = worksheet.get_all_values()
        headers = data.pop(0)
        batch = pd.DataFrame(data, columns=headers)
        batch_pandas = pd.DataFrame()
        # maintain a set of processed message IDs

        async def process_message(row):
            nonlocal batch_pandas
            msg_type = row["messages_type"]
            message_id = str(row["message_id"])
            sender = str(row["sender"])
            account_id = str(row["account_id"])
            # Check if the message ID is already processed
            if message_id not in processed_message_ids and (sender!=''):
                # Add the processed message ID to the set
                processed_message_ids.append(message_id)                
                
                if sender=='':
                    return
                if account_id!='106262272527366':
                    return

                message_data = {
                    'id': str(row["account_id"]),
                    'disp_phone_n': str(row["disp_phone_numb"]),
                    'phone_n_id': str(row["phone_number_id"]),
                    'contacts_name': str(row["profile_name"]),
                    'contacts_wa_id': str(row["contact_whatsapp_id"]),
                    'from': str(row["sender"]),
                    'wa_msg_id': str(row["message_id"]),
                    'timestamp': str(row["timestamp"]),
                    'type': str(row["messages_type"]),
                    'timestamp2': time.time()
                }
                
                if msg_type == 'text':
                    message_data['text_body'] = str(row["text_body"])
                elif msg_type == 'audio':
                    message_data['text_body'] = str(row["audio_id"])
                elif msg_type == 'image':
                    message_data['timestamp'] = str(row["img_caption"])
                    message_data['text_body'] = str(row["img_id"])
                elif msg_type == 'interactive':
                    msg_type_interactive = str(row["interactive_type"])
                    if msg_type_interactive == 'list_reply':
                        message_data['text_body'] = str(row["interactive_list_reply_id"])
                    elif msg_type_interactive == 'button_reply':
                        message_data['text_body'] = str(row["interactive_button_reply_id"])
                    else:
                        print('invalid message type 1')
                        return
                elif msg_type == 'button':
                    message_data['text_body'] = str(row["button_text"])

                else:                     
                    print('invalid message')
                    return

                df2 = pd.DataFrame(message_data, index=[0])
                batch_pandas = pd.concat([batch_pandas, df2]).reset_index(drop=True)
            else:
                return

        await asyncio.gather(*(process_message(batch.loc[i]) for i in range(len(batch))))

    return batch_pandas

async def update_conversations(batch_pandas):
    for i in range(len(batch_pandas)):  #almacenar mensajes en la conversación respectiva
        msg_from = str(batch_pandas['from'][i])
        contact_name = batch_pandas['contacts_name'][i]

        try:
            conv = pd.read_csv(root+'Data/Messages/'+msg_from+'.csv')
            #print('conv readed')
            if len(conv) < 25: #para que la data historica no sea tan grande (asegura que se hallan esperado varios mensajes hasta eliminar opciones mas antiguas)
                conv_msg_ids = conv['wa_msg_id'].values[-len(conv):]
            else:
                conv_msg_ids = conv['wa_msg_id'].values[-(25):]

            if str(batch_pandas['wa_msg_id'][i]) not in str(conv_msg_ids): #si no existe ese id..
                conv = pd.concat([conv,batch_pandas[i:i+1]]).reset_index(drop=True) #crear nuevo mensaje
                conv.to_csv(root+'Data/Messages/'+msg_from+'.csv',index=False) #almacenar mensaje
                print('nuevo mensaje - ',batch_pandas['from'][i])
        except:             
            print('Not Registered - ',batch_pandas['from'][i])
            await add_unknown(msg_from)
            conv = pd.read_csv(root+'Data/Messages/'+msg_from+'.csv')
            conv = pd.concat([conv,batch_pandas[i:i+1]]).reset_index(drop=True) #crear nuevo mensaje
            conv.loc[len(conv)] = ['01','nan','nan','AI' ,'nan' ,'AI','nan',time.time(),'First Template','PLANTILLA0',time.time()]
            #await send_plantilla_menu(conv,contact_number,language)
            conv.to_csv(root+'Data/Messages/'+msg_from+'.csv',index=False) #almacenar mensaje
            print('saved - ',batch_pandas['from'][i])

async def check_for_changes():
        global row_count
        global previous_row_count

        while True:
            try:
                print('check changes in messages_log')
                row_count = get_worksheet_row_count('1su6WIP6L3qtOR4zQNZoPr95SCgFrtWQru5F4mf1zRe8', 'Hoja 1', credentials)
                if row_count != previous_row_count:
                    batch = await process_batch()
                    print('update conversations')
                    await update_conversations(batch)  # Process the messages and update conversations

                    for _, row in batch.iterrows():
                        await incoming_messages.put(row)  # Add each new message to the incoming_messages queue
                        incoming_messages_semaphore.release()  # Increment the semaphore

                    previous_row_count = row_count
                await asyncio.sleep(3)

            except Exception as e:
                print('Error', e)
                print(traceback.format_exc())
                await asyncio.sleep(3)


# Worker function to process messages and interact with users simultaneously
async def worker(worker_id):
        while True:
            await incoming_messages_semaphore.acquire()  # Wait for the semaphore to be released
            try:
                message = await incoming_messages.get()
                message_id = message['wa_msg_id']

                if message_id not in processed_messages:
                    processed_messages.append(message_id)
                    worker_status[worker_id] = message_id
                    print(f'Worker {worker_id} - interact with user')
                    await interact_async(str(message['from']))  # Interact with the user
                    worker_status[worker_id] = None

            finally:
                incoming_messages_semaphore.release()  # Release the Semaphore

async def main_loop():
        global previous_row_count

        print('MAIN LOOP STARTED')
        previous_row_count = get_worksheet_row_count('1su6WIP6L3qtOR4zQNZoPr95SCgFrtWQru5F4mf1zRe8', 'Hoja 1', credentials)

        worker_tasks = [asyncio.create_task(worker(i)) for i in range(5)]
        check_for_changes_task = asyncio.create_task(check_for_changes())

        await asyncio.gather(*worker_tasks, check_for_changes_task)

print('main loop loaded')
asyncio.run(main_loop())