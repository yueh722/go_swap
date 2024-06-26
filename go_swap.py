
config_file = 'config.ini'

import sys
sys.path.append('./uniswap-python')
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from eth_typing.evm import Address, ChecksumAddress
from hexbytes import HexBytes
from decimal import Decimal

#from flask import Flask, request, jsonify
import os
import time
from time import sleep
import logging
import json
import urllib.request
import math
from web3 import Web3, HTTPProvider
from web3.contract import Contract

# for Telegram bot
import requests
import configparser

from datetime import datetime
from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
from telepot.loop import MessageLoop

import keyboard
import check_st as st


## 定義Uniswap流動性池（LP）類別 -----------
class Uniswap_LP:
    def __init__(self, nft_token_id, min_price, max_price, liquidity):
        self.nft_token_id = int(nft_token_id)
        self.min_price = Decimal(min_price)
        self.max_price = Decimal(max_price)
        self.liquidity = int(liquidity)


## 定義全域變數 -----------
version = 0.1

# initialize... (will get value from config file)
total_usdc_amount = None
keep_ETH_in_wallet = None
range_stop_loss = None
range_take_profit = None
ST_position: int = 0 # (external) strategy position (setting by signal)
LP_position: int = 0 # 0:No position, 1:Long(Uniswap), -1:(Aave+Uniswap), -2:(Aave, no liquidity on Uniswap)
uniswap_positions: List[int] = []
uniswap_newest_LP = Uniswap_LP(0, 0.0, 0.0, 0) # the last (newest) LP's nft_token_id, min_price, max_price, liquidity
# for signal reading
TIMER = None
ALERT_PATH = None
# for telegram bot
TG_TOKEN = None
TG_CHAT_ID = None
# for alert file's info.
Alert_Symbol = None
Alert_Strategy = None
Alert_Price = None
Alert_StopLoss = None
Alert_TakeProfit = None
Alert_Signature = None
Alert_Time = None

## 配置讀取
def get_config():
    global total_usdc_amount, keep_ETH_in_wallet, range_stop_loss, range_take_profit
    global TIMER, FEE_TIER, MAX_SLIPPAGE, AAVE_LTV, TX_DEADLINE, GAS_LIMIT, MAX_UINT_128, MAX_UINT_256, MAX_USDC_APPROVE_AMOUNT, MAX_WETH_APPROVE_AMOUNT
    global ALERT_PATH, TG_TOKEN, TG_CHAT_ID
    global infura_url 
    global ETH_address, weth_address, weth_abi, weth_decimals, usdc_address, usdc_abi, usdc_decimals
    global uniswapv3_eth_usdc_pool_address, uniswapv3_eth_usdc_pool_abi
    global uniswapv3_router2_address, uniswapv3_router2_abi
    global uniswapv3_posnft_address, uniswapv3_posnft_abi
    global aavev3_pool_address, aavev3_pool_abi
    global aavev3_weth_gateway_address, aavev3_weth_gateway_abi
    global wallet_address, wallet_private_key
    try:
        with open(config_file, 'r') as fp: #use this way to avoid not-close isseue
            config = configparser.ConfigParser()
            config.read_file(fp)
            fp.close()
    except:
        raise Exception('ERROR loading config file: ', config_file, '\n')

    # trade settings
    total_usdc_amount = Decimal(eval(config['TRADE']['total_usdc_amount'])) # in USDC
    keep_ETH_in_wallet = Decimal(eval(config['TRADE']['keep_ETH_in_wallet'])) # for gas fee
    range_stop_loss = Decimal(eval(config['TRADE']['range_stop_loss'])) #10.0 #4.0
    range_take_profit = Decimal(eval(config['TRADE']['range_take_profit'])) #10.0  #16.0
    # general settings
    TIMER = int(eval(config['GENERAL']['TIMER']))
    FEE_TIER = int(eval(config['GENERAL']['FEE_TIER'])) # 500 means 0.05% for uniswapv3_eth_usdc_pool_address
    MAX_SLIPPAGE = Decimal(eval(config['GENERAL']['MAX_SLIPPAGE'])) # 0.5 means 0.5% slippage
    AAVE_LTV = Decimal(eval(config['GENERAL']['AAVE_LTV'])) # Aave Loan-to-Value ratio
    TX_DEADLINE = int(eval(config['GENERAL']['TX_DEADLINE']))
    GAS_LIMIT = int(eval(config['GENERAL']['GAS_LIMIT']))
    MAX_UINT_128 = int(eval(config['GENERAL']['MAX_UINT_128']))
    MAX_UINT_256 = int(eval(config['GENERAL']['MAX_UINT_256']))
    MAX_USDC_APPROVE_AMOUNT = int(eval(config['GENERAL']['MAX_USDC_APPROVE_AMOUNT']))
    MAX_WETH_APPROVE_AMOUNT = int(eval(config['GENERAL']['MAX_WETH_APPROVE_AMOUNT']))
    # alert settings
    ALERT_PATH = config['ALERT']['alert_path'] 

    # telegram settings
    TG_TOKEN = config['TELEGRAM']['tg_token']
    TG_CHAT_ID = config['TELEGRAM']['tg_chat_id']
    if not(TG_TOKEN and TG_CHAT_ID):
        raise Exception('ERROR initializing TG_TOKEN or TG_CHAT_ID\n', 'TG_TOKEN: ', TG_TOKEN, '\n', 'TG_CHAT_ID:', TG_CHAT_ID, '\n')

    # get API keys for  Telegram
    # init Telegram bot
    #tg_apikey = config['API KEY']['telegram']
    #tg_bot = telepot.Bot(tg_apikey)
    #MessageLoop(tg_bot, {'chat': on_chat_message}).run_as_thread()

    infura_url = config['URL']['infura']

    # 設定代幣及pool地址
    ETH_address = config['ADDRESS']['eth_address']  

    # WETH token contract
    weth_address = config['ADDRESS']['weth_address'] # WETH token0
    weth_abi = config['ABI']['weth_abi'] 
    weth_decimals = int(config["DECIMALS"]['weth']) # ETH has 18 decimals

    # USDC token contract
    usdc_address = config['ADDRESS']['usdc_address'] # USDC.e token1
    usdc_abi = config['ABI']['usdc_abi'] 
    usdc_decimals = int(config["DECIMALS"]['usdc']) # USDC has 6 decimals

    # Uniswap V3 USDC.e/ETH 0.5% pool contract
    uniswapv3_eth_usdc_pool_address = config['ADDRESS']['uniswapv3_eth_usdc_pool_address'] 
    uniswapv3_eth_usdc_pool_abi = config['ABI']['uniswapv3_eth_usdc_pool_abi'] 

    # Uniswap V3 Router contract
    uniswapv3_router2_address = config['ADDRESS']['uniswapv3_router2_address'] 
    uniswapv3_router2_abi = config['ABI']['uniswapv3_router2_abi']

    # Uniswap V3 positions NFT contract
    uniswapv3_posnft_address = config['ADDRESS']['uniswapv3_posnft_address'] 
    uniswapv3_posnft_abi = config['ABI']['uniswapv3_posnft_abi']

    # Aave V3 pool contract
    aavev3_pool_address = config['ADDRESS']['aavev3_pool_address'] 
    aavev3_pool_abi = config['ABI']['aavev3_pool_abi']

    # Aave V3 pool contract
    aavev3_weth_gateway_address = config['ADDRESS']['aavev3_pool_address'] 
    aavev3_weth_gateway_abi = config['ABI']['aavev3_weth_gateway_abi']

    # wallet
    wallet_address = config['ADDRESS']['wallet_airdrop']
    wallet_private_key = config['PRIVATE KEY']['wallet_airdrop']

    return


## 發送Telegram 訊息
def send_message(msg):
    # 1. print in lost
    print(msg)
    # 2. sned message to TG
    global tgbot
    if msg is None:
        print("No message!")
        return
    while (True):
        try:
            tgurl = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage?chat_id={TG_CHAT_ID}&text=' + msg
            requests.get(tgurl)
            break
        except:
            sleep(TIMER)
            #raise Exception('ERROR sending message to TG\n')

    return


## 策略解讀
def handle_alert(alert_string):
    global Alert_Symbol, Alert_Strategy, Alert_Price, Alert_StopLoss, Alert_TakeProfit, Alert_Signature, Alert_Time
    global range_stop_loss, range_take_profit

    # 1. reading Alert info.
    alert_msg = ""
    config = configparser.ConfigParser()
    config.read_string(alert_string)
    Alert_Symbol = config['ALERT']['Symbol']
    Alert_Strategy = config['ALERT']['Strategy']
    Alert_Price = Decimal(eval(config['ALERT']['Price']))
    Alert_StopLoss = Decimal(eval(config['ALERT']['StopLoss']))
    Alert_TakeProfit = Decimal(eval(config['ALERT']['TakeProfit']))
    Alert_Signature = config['ALERT']['Signature']
    Alert_Time = config['ALERT']['Time']
    if (Alert_StopLoss):
        range_stop_loss = Alert_StopLoss
    if (Alert_TakeProfit):
        range_take_profit = Alert_TakeProfit

    # 2. handling Alert
    if ("Long by ST_26738832" in Alert_Signature):
        if st.go_signal(1) == 0:
            alert_msg += "Long\n"
    elif ("Short by ST_26738832" in Alert_Signature):
        if st.go_signal(-1) == 0:
            alert_msg += "Short\n"
    elif ("Flat by ST_26738832" in Alert_Signature):
        if st.go_signal(0) == 0:
            alert_msg += "Flat\n"
    else:
        alert_msg += "Unknown signature: " + Alert_Signature + "\n"
        send_message(alert_msg)
        return

    # 3. show alert message
    end_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    alert_msg += (end_time + "\n====================\n")
    send_message(alert_msg)

    # 4. update LP_position, uniswap_positions and show status
    status_msg = st.get_status() # get wallet balance and price info.
    status_msg += st.update_LP_position_and_uniswap_positions() # update LP_position, uniswap_positions
    send_message(status_msg)


## -------------------------------------------

def main():
    get_config()
    msg = ""
    
    
    # check approval 
    #need_check_allowance = True
    need_check_allowance = False
    st.do_approval(need_check_allowance)
    
    
    msg += "Init ok!\n"
    send_message(msg)
    
    # update and show status
    msg = st.get_status() # get wallet balance and price info.
    msg += st.update_LP_position_and_uniswap_positions() # update LP_position, uniswap_positions
    send_message(msg)
    msg = ""
    
    
    # list all files in the Alert folder
    if ALERT_PATH:
        before = os.listdir(ALERT_PATH)
        #msg += "Existing files:\n" + ', '.join(before) + "\n"
    else:
        raise Exception('ERROR initializing ALERT_PATH: ', ALERT_PATH, '\n')

    print('Press "Q" or "q" to exit the loop.')

    # read each new added file
    while True:
        if keyboard.is_pressed('Q') or keyboard.is_pressed('q'):
            print("Exiting loop...")
            break

        sleep(TIMER)
    
        # 循環監控指定目錄
        after = os.listdir(ALERT_PATH)
        added = [file for file in after if file not in before]
    
        # 檢測到有新文件
        if len(added) > 0:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = ("====================\n" + "Start Time : " + start_time + "\n")
            msg += ', '.join(added) + "\n"
            for file in added:
                # read the alert file
                fileFullname = ALERT_PATH + file
                alert_string = ""
                try:
                    with open(fileFullname, 'r') as fp:
                        alert_string = fp.read()
                        fp.close()
                except:
                    raise Exception('ERROR loading Alert file: ', fileFullname, '\n')
                # show and handle alert
                msg += alert_string + "\n"
                send_message(msg)
                handle_alert(alert_string)
        before = after

if __name__ == "__main__":
    main()
