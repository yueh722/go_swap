
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

from flask import Flask, request, jsonify
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

class Uniswap_LP:
    def __init__(self, nft_token_id, min_price, max_price, liquidity):
        self.nft_token_id = int(nft_token_id)
        self.min_price = Decimal(min_price)
        self.max_price = Decimal(max_price)
        self.liquidity = int(liquidity)

# global variables -------------------------
#version = 0.1

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


# get config
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


# send to TG 
def send_message(msg):
    # 1. print in local
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

# useful functions for all
# 解決型別強制轉換，讓 function match type 時不會出現如下 error
# ... Found 1 function(s) with the name `mint`: ['mint((address,address,uint24,int24,int24,uint256,uint256,uint256,uint256,address,uint256))'] Function invocation failed due to improper number of arguments.
# https://steam.oxxostudio.tw/category/python/numpy/numpy-dtype.html
import numpy as np
#dt = np.dtype([('a', 'u4')])
#dt = np.dtype('uint32')
#a = np.array([(1.1)], dtype=np.ulonglong)
#a.dtype = np.ctypeslib.as_ctypes(np.uint256)
#print(a[0])    # [('1.1', 2.2,  True) ('1.1', 2.2,  True)]
#print(a.dtype)    # [('1.1', 2.2,  True) ('1.1', 2.2,  True)]
int16 ='i2'
int32 ='i4'
uint16 ='u2'
uint32 ='u4'
uint64 ='u8'
def change_type(input_data, type):
    np_array = np.array([input_data])
    new_array = np_array.astype(type)
    return new_array[0]
def usdc_in_human_to_wei(usdc_amount_in_human):
    return(int(Decimal(usdc_amount_in_human) * Decimal(10 ** usdc_decimals)))  # USDC 的精度通常是 6
def eth_in_human_to_wei(eth_amount_in_human):
    return(int(Decimal(eth_amount_in_human) * Decimal(10 ** weth_decimals)))  # ETH 的精度通常是 18
def usdc_in_wei_to_human(usdc_amount_in_human):
    return(Decimal(usdc_amount_in_human) / Decimal(10 ** usdc_decimals))  # USDC 的精度通常是 6
def eth_in_wei_to_human(eth_amount_in_human):
    return(Decimal(eth_amount_in_human) / Decimal(10 ** weth_decimals))  # ETH 的精度通常是 18


# -------------------------------
# useful functions for uniswap V3
# Convert Uniswap v3 tick to a price (i.e. the ratio between the amounts of tokens: token1/token0)
TICK_BASE = 1.0001
def tick_to_price(tick):
    return Decimal(TICK_BASE ** tick)
def price_to_tick(price):
    return Decimal(math.floor(math.log(price, TICK_BASE)))

# Not all ticks can be initialized. Tick spacing is determined by the pool's fee tier.
def fee_tier_to_tick_spacing(fee_tier):
    return {
        100: 1,
        500: 10,
        3000: 60,
        10000: 200
    }.get(fee_tier, 60)

# https://github.com/Uniswap/v3-core/blob/v1.0.0/contracts/libraries/TickMath.sol#L8-L11
MIN_TICK = -887272
MAX_TICK = -MIN_TICK
def get_nearest_usable_tick(tick, tick_spacing):
    #min_tick, max_tick = get_default_tick_range(min_price, max_price, tick_spacing)
    #assert min_tick <= tick <= max_tick, "Tick out of bound"
    rounded = round(tick / tick_spacing) * tick_spacing
    if rounded < MIN_TICK:
        return rounded + tick_spacing
    elif rounded > MAX_TICK:
        return rounded - tick_spacing
    return rounded

def calculate_liquidity_amounts(eth_price, lower_price_bound, upper_price_bound, total_usdc):
    """
    Calculate the amounts of ETH and USDC to provide liquidity in a Uniswap V3 pool.
    :param eth_price: Current price of ETH in USDC.
    :param lower_price_bound: Lower price bound of the price range.
    :param upper_price_bound: Upper price bound of the price range.
    :param total_usdc: Total amount of USDC to invest.
    :return: Amounts of ETH and USDC to provide.
    """
    if not (lower_price_bound <= eth_price <= upper_price_bound):
        raise ValueError("Current price is out of the specified range.")
    # Calculate the square roots for the prices
    sqrt_lower = math.sqrt(lower_price_bound)
    sqrt_upper = math.sqrt(upper_price_bound)
    sqrt_price = math.sqrt(eth_price)
    # Calculate the amounts of ETH and USDC
    eth_amount = total_usdc * Decimal(sqrt_upper - sqrt_price) / Decimal(sqrt_upper - sqrt_lower) / Decimal(eth_price)
    usdc_amount = total_usdc * Decimal(sqrt_price - sqrt_lower) / Decimal(sqrt_upper - sqrt_lower)
    return eth_amount, usdc_amount

# Three ways to get current price
def get_price():
    # get price by using uniswapV3 contract directly
    _, price_tick, *_ = uniswapv3_eth_usdc_pool_contract.functions.slot0().call()
    price = tick_to_price(price_tick)
    return (Decimal(price) / Decimal(10 ** (usdc_decimals - weth_decimals)))
''' other ways...
def get_price_defi():
    # get price by using eth_defi.uniswap_v3 module => TOO SLOW!!!
    return get_onchain_price(web3, uniswapv3_eth_usdc_pool_address) # another way
def get_price_uni():
    # get price by using uniswap-python module => 不準!!!
    return uniswap_py.get_raw_price(weth_address, usdc_address, fee=FEE_TIER) # another way
'''

def get_price_tick():
    # get price by using uniswapV3 contract directly
    _, price_tick, *_ = uniswapv3_eth_usdc_pool_contract.functions.slot0().call()
    price = tick_to_price(price_tick)
    return (Decimal(price) / Decimal(10 ** (usdc_decimals - weth_decimals))), price_tick

def calculate_range_ticks(price_tick, min_price, max_price):
    # 範圍設定
    min_tick = price_to_tick(min_price * Decimal(10 ** (usdc_decimals - weth_decimals))) #tick / price * min_price #price_to_tick(min_price)
    max_tick = price_to_tick(max_price * Decimal(10 ** (usdc_decimals - weth_decimals))) #tick / price * max_price #price_to_tick(max_price)
    # 調整範圍 by tick 及 tick_spacing
    #L = uniswapv3_eth_usdc_pool_contract.functions.liquidity().call()
    tick_spacing = fee_tier_to_tick_spacing(FEE_TIER)
    adjusted_min_tick = get_nearest_usable_tick(min_tick, tick_spacing)
    adjusted_max_tick = get_nearest_usable_tick(max_tick, tick_spacing)
    if (adjusted_min_tick > price_tick):
        adjusted_min_tick -= tick_spacing
    if (adjusted_max_tick < price_tick):
        adjusted_max_tick += tick_spacing
    adjusted_min_price_in_human = tick_to_price(adjusted_min_tick) / Decimal(10 ** (usdc_decimals - weth_decimals))
    adjusted_max_price_in_human = tick_to_price(adjusted_max_tick) / Decimal(10 ** (usdc_decimals - weth_decimals))
    #print({adjusted_min_tick}, {adjusted_max_tick})
    #print({adjusted_min_price_in_human}, {adjusted_max_price_in_human})
    return adjusted_min_tick, adjusted_max_tick, adjusted_min_price_in_human, adjusted_max_price_in_human

def get_aave_pool_status():
    totalCollateralBase, totalDebtBase, *_ = aavev3_pool_contract.functions.getUserAccountData(wallet_address).call()
    collateral_in_wei =  usdc_in_human_to_wei(Decimal(totalCollateralBase) / Decimal(10 ** 8))
    debt_in_wei =  usdc_in_human_to_wei(Decimal(totalDebtBase) / Decimal(10 ** 8))
    return collateral_in_wei, debt_in_wei

# get status info.
def get_status():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = ("*** " + start_time + " status ***\n")
    # General status
    balance_usdc_in_wei = usdc_contract.functions.balanceOf(wallet_address).call()
    balance_usdc_in_human = usdc_in_wei_to_human(balance_usdc_in_wei)
    balance_weth_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
    balance_weth_in_human = eth_in_wei_to_human(balance_weth_in_wei)
    balance_eth_in_wei = web3.eth.get_balance(wallet_address)
    balance_eth_in_human = eth_in_wei_to_human(balance_eth_in_wei)
    status += "* Wallet: {:.3f} USDC + {:.3f} WETH + {:.3f} ETH\n".format(balance_usdc_in_human, balance_weth_in_human, balance_eth_in_human)
    price = get_price()
    status += "* Price: {:.2f}\n".format(price)
    return status

# update LP_position, uniswap_positions, and uniswap_newest_LP
# check all positions in Uniswap and all CDP in Aave
''' Sample: positions(1019872)
nonce        uint96 : 0
operator    address : 0x0000000000000000000000000000000000000000
token0      address : 0x82aF49447D8a07e3bd95BD0d56f35241523fBab1
token1      address : 0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8
fee          uint24 : 500
tickLower     int24 : -198970
tickUpper     int24 : -198910
liquidity   uint128 : 539407368785800
feeGrowthInside0LastX128 uint256 : 115792089237316195423570985008687907654354900095754920212762857153196536567606
feeGrowthInside1LastX128 uint256 : 115792089237316195423570985008687907853269984072305937864239639501606588612829
tokensOwed0 uint128 : 0
tokensOwed1 uint128 : 0
'''
def update_LP_position_and_uniswap_positions():
    print("update_LP_position_and_uniswap_positions()")
    # 1. having Uniswap LP?
    global LP_position, uniswap_positions, uniswap_newest_LP
    number_of_positions = uniswapv3_posnft_contract.functions.balanceOf(wallet_address).call()
    LP_position = 0
    uniswap_positions = []
    if number_of_positions > 0:
        for idx in range(number_of_positions):
            position = (uniswapv3_posnft_contract.functions.tokenOfOwnerByIndex(wallet_address, idx).call())
            uniswap_positions.append(position)
            # check if any alive LP
            uniswap_newest_LP.nft_token_id = uniswap_positions[idx]
            _, _, _, _, _, tickLower, tickUpper, uniswap_newest_LP.liquidity, *_ = uniswapv3_posnft_contract.functions.positions(uniswap_newest_LP.nft_token_id).call()
            uniswap_newest_LP.min_price = tick_to_price(tickLower) / Decimal(10 ** (usdc_decimals - weth_decimals))
            uniswap_newest_LP.max_price = tick_to_price(tickUpper) / Decimal(10 ** (usdc_decimals - weth_decimals))
            if (uniswap_newest_LP.liquidity > 0):
                LP_position = 1
    # 2. having Aave CDP?
    collateral_in_wei, debt_in_wei = get_aave_pool_status()
    if (collateral_in_wei > 0 or debt_in_wei > 0):
        # 會覆蓋掉 case "current_position = 1"
        if (LP_position > 0):
            LP_position = -1 
        elif (LP_position == 0):
            # 空單交易做了一半: 放了 Aave, 但 Uniswap 沒有部位 (可能是程式 error 造成, 或重進空單時正在調整部位)
            LP_position = -2
        #print("- Aave: {:.2f} USD = {:.2f} - {:.2f}".format(usdc_in_wei_to_human(collateral_in_wei - debt_in_wei), usdc_in_wei_to_human(collateral_in_wei), usdc_in_wei_to_human(debt_in_wei)))

    # Compile position status
    status = ""
    if (LP_position != 0):
        if (LP_position > 0):
            status += "* Long LP ({})\n". format(LP_position)
        elif (LP_position < 0):
            status += "* Short LP ({})\n". format(LP_position)
        # 1. Uniswap status
        if (uniswap_newest_LP.min_price != 0):
            status += "* Uniswap: ({:.2f} ~ {:.2f})\n".format(uniswap_newest_LP.min_price, uniswap_newest_LP.max_price)
        # 2. Aave status
        #collateral_in_wei, debt_in_wei = get_aave_pool_status()
        if (collateral_in_wei > 0 or debt_in_wei > 0):
            status += "* Aave: {:.2f} USD = {:.2f} - {:.2f}\n".format(usdc_in_wei_to_human(collateral_in_wei - debt_in_wei), usdc_in_wei_to_human(collateral_in_wei), usdc_in_wei_to_human(debt_in_wei))
    return status

# fuctions with tx --------------------------------------------------------------------------
# check approve for usdc and weth
def check_allowance(amount_in_wei, token_contract, contract_checksum_address) -> HexBytes:
    # 检查是否已授权足够的 token
    allowance = token_contract.functions.allowance(wallet_checksum_address, contract_checksum_address).call()
    if allowance < amount_in_wei:
        # 一次 apporve 很大的數量
        max_amount_to_approve = web3.to_wei(amount_in_wei, 'ether')
        approve_tx = token_contract.functions.approve(
            contract_checksum_address,
            max_amount_to_approve
        ).build_transaction({
            'from': wallet_checksum_address,
            'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
            'gas': 2000000,  # Gas 设置可能需要调整
            #'gasPrice': web3.toWei('50', 'gwei')
            'type': '0x2', # EIP-1559
        })
        signed_tx = web3.eth.account.sign_transaction(approve_tx, wallet_private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt.status == 1
        print(f"*** Approve TX Hash: {tx_hash.hex()}")
        return tx_hash
    return 0

def check_balance_usdc(needed_usdc_in_human) -> bool:
    balance_usdc_in_wei = usdc_contract.functions.balanceOf(wallet_address).call()
    balance_usdc_in_human = usdc_in_wei_to_human(balance_usdc_in_wei)
    return (balance_usdc_in_human >= needed_usdc_in_human) 
def check_balance_weth(needed_weth_in_human) -> bool:
    balance_weth_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
    balance_weth_in_human = eth_in_wei_to_human(balance_weth_in_wei)
    return (balance_weth_in_human >= needed_weth_in_human) 
def check_balance_ETH(needed_eth_in_human) -> bool:
    balance_eth_in_wei = web3.eth.get_balance(wallet_address)
    balance_eth_in_human = eth_in_wei_to_human(balance_eth_in_wei)
    return (balance_eth_in_human >= needed_eth_in_human) 

# Uniswap --------------------------------------------------
# for swap and add liquidity
def build_swap_tx(tokenIn, tokenOut, amountIn_in_wei, min_amountOut_in_wei):
    swap_tx = uniswapv3_router2_contract.functions.exactInputSingle({
        'tokenIn': tokenIn,
        'tokenOut': tokenOut,
        'fee': FEE_TIER,
        'recipient': wallet_address,
        'deadline': TX_DEADLINE,
        'amountIn': amountIn_in_wei,
        'amountOutMinimum': min_amountOut_in_wei,
        'sqrtPriceLimitX96': 0
    }).build_transaction({
        'from': wallet_address,
        'gas': GAS_LIMIT,  # Gas 设置可能需要调整
        #'gasPrice': web3.to_wei(5, 'gwei'), 
        'type': '0x2', # EIP-1559
        'nonce': web3.eth.get_transaction_count(wallet_address),
        'value': 0
    })
    return swap_tx
def swap_usdc2weth(usdc_amount_in_human, need_balance_check: Optional [bool] = False) -> HexBytes:
    if (need_balance_check and check_balance_usdc(usdc_amount_in_human) == False): 
        # 要檢查錢包內 USDC => USDC 不夠
        print(f"Error Message swap_usdc2weth(): USDC in wallet < needed {usdc_amount_in_human}!")
        return 0
    # 已检查過是否已授权足够的 USDC
    # swapping...
    print(f"Let {usdc_amount_in_human} USDC -> WETH...")
    usdc_amount_to_swap_in_wei = usdc_in_human_to_wei(usdc_amount_in_human)
    #tx_hash = uniswap_py.make_trade(usdc_address, weth_address, usdc_amount_to_swap_in_wei, fee=FEE_TIER)
    min_amount_in_wei = usdc_in_human_to_wei(Decimal(usdc_amount_in_human) / Decimal(get_price()) * Decimal((100 - MAX_SLIPPAGE)/100))
    swap_tx = build_swap_tx(usdc_address, weth_address, usdc_amount_to_swap_in_wei, min_amount_in_wei)
    signed_tx = web3.eth.account.sign_transaction(swap_tx, wallet_private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** Swap TX Hash: {tx_hash.hex()}")
    return tx_hash
def swap_usdc2weth_output(weth_amount_in_human) -> HexBytes:
    # 已检查過是否已授权足够的 USDC
    # swapping...
    print(f"Let USDC -> {weth_amount_in_human} WETH...")
    weth_amount_in_wei = eth_in_human_to_wei(weth_amount_in_human)
    #tx_hash = uniswap_py.make_trade_output(usdc_address, weth_address, weth_amount_in_human_in_wei, fee=FEE_TIER)
    min_amount_in_wei = weth_amount_in_wei
    usdc_amount_to_swap_in_wei = usdc_in_human_to_wei(Decimal(weth_amount_in_human) * Decimal(get_price()) * Decimal((100 + MAX_SLIPPAGE)/100))
    swap_tx = build_swap_tx(usdc_address, weth_address, usdc_amount_to_swap_in_wei, min_amount_in_wei)
    signed_tx = web3.eth.account.sign_transaction(swap_tx, wallet_private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** Swap TX Hash: {tx_hash.hex()}")
    return tx_hash

def swap_weth2usdc(weth_amount_in_human, need_balance_check: Optional [bool] = False) -> HexBytes:
    if (need_balance_check and check_balance_weth(weth_amount_in_human) == False): 
        # 要檢查錢包內 WETH => WTH 不夠
        print(f"Error Message swap_weth2usdc(): WETH in wallet < needed {weth_amount_in_human}!")
        return 0
    # 已检查過是否已授权足够的 WETH
    # swapping...
    print(f"Let {weth_amount_in_human} WETH -> USDC...")
    weth_amount_to_swap_in_wei = eth_in_human_to_wei(weth_amount_in_human)
    #tx_hash = uniswap_py.make_trade(weth_address, usdc_address, weth_amount_to_swap_in_wei, fee=FEE_TIER)
    min_amount_in_wei = usdc_in_human_to_wei(Decimal(weth_amount_in_human) * Decimal(get_price()) * Decimal((100 - MAX_SLIPPAGE)/100))
    swap_tx = build_swap_tx(weth_address, usdc_address, weth_amount_to_swap_in_wei, min_amount_in_wei)
    signed_tx = web3.eth.account.sign_transaction(swap_tx, wallet_private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** Swap TX Hash: {tx_hash.hex()}")
    return tx_hash
'''
def swap_usdc2ETH(usdc_amount_in_human, need_balance_check: Optional [bool] = False) -> HexBytes:
    if (need_balance_check and check_balance_usdc(usdc_amount_in_human) == False): 
        # 要檢查錢包內 USDC => USDC 不夠
        print(f"Error Message swap_usdc2ETH(): USDC in wallet < needed {usdc_amount_in_human}!")
        return 0
    # 已检查過是否已授权足够的 USDC
    # swapping...
    print(f"Let {usdc_amount_in_human} USDC -> ETH")
    usdc_amount_to_swap_in_wei = usdc_in_human_to_wei(usdc_amount_in_human)
    tx_hash = uniswap_py.make_trade(usdc_address, ETH_address, usdc_amount_to_swap_in_wei, fee=FEE_TIER)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** Swap TX Hash: {tx_hash.hex()}")
    return tx_hash

def swap_ETH2usdc(eth_amount_in_human, need_balance_check: Optional [bool] = False) -> HexBytes:
    if (need_balance_check and check_balance_ETH(eth_amount_in_human) == False): 
        # 要檢查錢包內 ETH => ETH 不夠
        print(f"Error Message swap_ETH2usdc(): ETH in wallet < needed {eth_amount_in_human}!")
        return 0
    # swapping...
    print(f"Let {eth_amount_in_human} ETH -> USDC...")
    eth_amount_to_swap_in_wei = eth_in_human_to_wei(eth_amount_in_human)
    tx_hash = uniswap_py.make_trade(ETH_address, usdc_address, eth_amount_to_swap_in_wei, fee=FEE_TIER)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** Swap TX Hash: {tx_hash.hex()}")
    return tx_hash
'''
def wrap_ETH(eth_amount_in_human, need_balance_check: Optional [bool] = False) -> HexBytes:
    if (need_balance_check and check_balance_ETH(eth_amount_in_human) == False): 
        # 要檢查錢包內 ETH => ETH 不夠
        print(f"Error Message wrap_ETH(): ETH in wallet < needed {eth_amount_in_human}!")
        return 0
    print(f"Let {eth_amount_in_human} ETH wrap...")
    eth_amount_in_wei = eth_in_human_to_wei(eth_amount_in_human)    
    wrap_tx = weth_contract.functions.deposit().build_transaction({
        'from': wallet_address,
        'value': eth_amount_in_wei, 
        'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
        'gas': GAS_LIMIT,  # Gas 设置可能需要调整
        #'gasPrice': web3.to_wei('5', 'gwei'), 
        'type': '0x2', # EIP-1559
    })
    signed_wrap_tx = web3.eth.account.sign_transaction(wrap_tx, wallet_private_key)
    wrap_tx_hash = web3.eth.send_raw_transaction(signed_wrap_tx.rawTransaction)
    print(f"*** Wrap TX Hash: {wrap_tx_hash.hex()}")
    return wrap_tx_hash

def wrap_all_ETH():
    # 錢包只保留 keep_ETH ETH for gas fee, 多餘的先 wrap 成 WETH
    balance_eth_in_wei = web3.eth.get_balance(wallet_address)
    balance_eth = eth_in_wei_to_human(balance_eth_in_wei) - Decimal(keep_ETH_in_wallet) # 保留 keep_ETH for gas fee
    if (balance_eth > 0.005): 
        print("Let all ETH -> WETH...")
        wrap_tx_hash = wrap_ETH(balance_eth)

def swap_all_weth():
    # 將所有 WETH 換成 USDC
    balance_weth_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
    balance_weth = eth_in_wei_to_human(balance_weth_in_wei)
    if (balance_weth > 0.005):
        print("Let all WETH -> USDC...")
        sell_tx_hash = swap_weth2usdc(Decimal(balance_weth))

def add_LP(weth_amount_in_wei, usdc_amount_in_wei, adjusted_min_tick, adjusted_max_tick, need_balance_check: Optional [bool] = False) -> HexBytes:
    # 放流動池
    print(f"add_LP({weth_amount_in_wei}, {usdc_amount_in_wei}, {adjusted_min_tick}, {adjusted_max_tick}, {need_balance_check})...")
    # 檢查錢包餘額
    weth_amount_in_human = eth_in_wei_to_human(weth_amount_in_wei)
    if (need_balance_check and check_balance_weth(weth_amount_in_human) == False): 
        # 要檢查錢包內 WETH => WTH 不夠
        print(f"Error Message add_LP(): WETH in wallet < needed {weth_amount_in_human}!")
        return 0
    usdc_amount_in_human = usdc_in_wei_to_human(usdc_amount_in_wei)
    if (need_balance_check and check_balance_usdc(usdc_amount_in_human) == False): 
        # 要檢查錢包內 USDC => USDC 不夠
        print(f"Error Message add_LP(): USDC in wallet < needed {usdc_amount_in_human}!")
        return 0

    # 執行 add LP
    fee = int(change_type(FEE_TIER, uint16))
    min_tick = int(change_type(adjusted_min_tick, int32))
    max_tick = int(change_type(adjusted_max_tick, int32))
    weth_amount_in_wei = weth_amount_in_wei- 1  # 沒有 -1 會有 error
    #weth_amount_in_wei = int(change_type(weth_amount_in_wei-1, 'uint')) # 沒有 -1 會有 error
    #usdc_amount_in_wei = int(change_type(usdc_amount_in_wei, 'uint'))
    amount0Min = int(0)
    amount1Min = int(0)
    deadline = TX_DEADLINE #int(18446744073709551616)
    mint_tx = uniswapv3_posnft_contract.functions.mint(
        (
            weth_checksum_address,
            usdc_checksum_address,
            fee,
            min_tick, 
            max_tick, 
            weth_amount_in_wei, 
            usdc_amount_in_wei, 
            amount0Min, 
            amount1Min, 
            wallet_checksum_address,
            deadline,
        ),
    ).build_transaction({
        'from': wallet_checksum_address,
        'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
        'gas': GAS_LIMIT,  # Gas 设置可能需要调整
        #'gasPrice': web3.to_wei('5', 'gwei'), 
        'type': '0x2', # EIP-1559
    })

    is_send = False
    if(is_send == True):
        print("after add LP build_transaction")
        signed_tx = web3.eth.account.sign_transaction(mint_tx, wallet_private_key)
        #print("after add LP sign_transaction")
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        #print("after add LP send_raw_transaction")
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt.status == 1
        print(f"*** AddLP TX Hesh: {tx_hash.hex()}")
        return tx_hash
    else :
        print(f"不送出 mint_tx = {mint_tx} \n")


def close_LP(nft_token_id):
    print(f"close_LP({nft_token_id})...")
    # Two steps: 1) decrease LP 2) collect fees
    # 0. get LP liquidity
    ''' Sample: positions(1019872)
    nonce        uint96 : 0
    operator    address : 0x0000000000000000000000000000000000000000
    token0      address : 0x82aF49447D8a07e3bd95BD0d56f35241523fBab1
    token1      address : 0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8
    fee          uint24 : 500
    tickLower     int24 : -198970
    tickUpper     int24 : -198910
    liquidity   uint128 : 539407368785800
    feeGrowthInside0LastX128 uint256 : 115792089237316195423570985008687907654354900095754920212762857153196536567606
    feeGrowthInside1LastX128 uint256 : 115792089237316195423570985008687907853269984072305937864239639501606588612829
    tokensOwed0 uint128 : 0
    tokensOwed1 uint128 : 0
    '''
    _, _, _, _, _, _, _, liquidity, *_ = uniswapv3_posnft_contract.functions.positions(nft_token_id).call()
    #print(f"Position {nft_token_id} has liquidity {liquidity}.")
    if (liquidity <= 0):
        print(f"No need to close position {nft_token_id} with 0 liquidity")
        return

    # 1. decrease LP
    tx_remove_liquidity = uniswapv3_posnft_contract.functions.decreaseLiquidity(
        (nft_token_id, liquidity, int(0), int(0), TX_DEADLINE)
    ).build_transaction({
        'from': wallet_checksum_address,
        'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
        'gas': GAS_LIMIT,  # Gas 设置可能需要调整
        #'gasPrice': web3.to_wei('5', 'gwei'), 
        'type': '0x2', # EIP-1559
    })
    signed_tx = web3.eth.account.sign_transaction(tx_remove_liquidity, wallet_private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** DecreaseLP TX Hesh: {tx_hash.hex()}")

    # 2. collect fees
    tx_collect_fees = uniswapv3_posnft_contract.functions.collect(
        (nft_token_id, wallet_address, MAX_UINT_128, MAX_UINT_128)
    ).build_transaction({
        'from': wallet_checksum_address,
        'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
        'gas': GAS_LIMIT,  # Gas 设置可能需要调整
        #'gasPrice': web3.to_wei('5', 'gwei'), 
        'type': '0x2', # EIP-1559
    })
    signed_tx = web3.eth.account.sign_transaction(tx_collect_fees, wallet_private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** CollectFees TX Hesh: {tx_hash.hex()}")
    return tx_hash

    # R: No need to burn?
    tx_burn = uniswapv3_posnft_contract.functions.burn(nft_token_id
    ).build_transaction({
        'from': wallet_checksum_address,
        'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
        'gas': GAS_LIMIT,  # Gas 设置可能需要调整
        #'gasPrice': web3.to_wei('5', 'gwei'), 
        'type': '0x2', # EIP-1559
    })
    signed_tx = web3.eth.account.sign_transaction(tx_burn, wallet_private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status == 1
    print(f"*** Burn TX Hesh: {tx_hash.hex()}")
    return tx_receipt

def close_LP_all():
    print(f"close_LP_all()...")
    global uniswap_newest_LP
    # R: 節省時間, 只做最新的一個
    close_LP(uniswap_newest_LP.nft_token_id)
    '''
    global uniswap_positions
    number_of_positions = len(uniswap_positions)
    for idx in range(number_of_positions):
        nft_token_id = uniswap_positions[idx]
        close_LP(nft_token_id)
    '''

# Aave --------------------------------------------------
# for make/close CDP for short
# 要借出等值 needed_usdc USDC 的 WETH 出來, 預設是錢包內只有 USDC (已沒有 WETH)
def make_CDP(needed_usdc): 
    print(f"make_CDP({needed_usdc})...")
    # followings are all usdc based
    collateral_in_wei, debt_in_wei = get_aave_pool_status() # 已經借了 debt_in_wei 等值的 WETH, 還需 (needed_usdc - debt)
    print(f"Aave Collateral {collateral_in_wei}, Debt: {debt_in_wei} wei USDC")
    needed_debt_in_wei = usdc_in_human_to_wei(needed_usdc) - debt_in_wei
    if (needed_debt_in_wei <= 0): # 已借足夠, 無需動作 (額外做的確保動作 不一定需要)
        return 0
    needed_collateral_in_wei = Decimal(needed_debt_in_wei + debt_in_wei) / Decimal(AAVE_LTV) - collateral_in_wei
    needed_collateral_in_human = int(usdc_in_wei_to_human(needed_collateral_in_wei))
    if (needed_collateral_in_human > 20): # 債資比 > AAVE_LTV (0.7)
        # collateral 不夠，需要再放 >20 USDC
        print(f"Need to deposit {needed_collateral_in_human} USDC")
        # 0. 檢查錢包內 USDC 是否足夠, 預設是錢包內只有 USDC (已沒有 WETH)
        if (check_balance_usdc(needed_collateral_in_human) == False):
            # 目前錢包中 USDC 不夠 => 將所有 WETH 換成 USDC
            swap_all_weth() # 將 WETH 清空, 為了生出 make_CDP 需要的 USDC
            if (check_balance_usdc(needed_collateral_in_human) == False): # 換完再 check 一次
                # USDC 不夠 => 不動作 (錢包內全持有 USDC)
                print(f"Error Message: USDC in wallet < needed {needed_collateral_in_human} USDC!")
                return 0
        
        # 1. 存入 USDC to Aave v3 pool
        deposit_tx = aavev3_pool_contract.functions.deposit(
            usdc_address, 
            int(needed_collateral_in_wei),
            wallet_address, 
            int(0)
        ).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
            'gas': GAS_LIMIT,  # Gas 设置可能需要调整
            #'gasPrice': web3.toWei('50', 'gwei')
            'type': '0x2', # EIP-1559
        })
        signed_deposit_tx = web3.eth.account.sign_transaction(deposit_tx, wallet_private_key)
        deposit_tx_hash = web3.eth.send_raw_transaction(signed_deposit_tx.rawTransaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(deposit_tx_hash)
        assert tx_receipt.status == 1
        print(f"*** Deposit TX Hash: {deposit_tx_hash.hex()}")
        #return deposit_tx_hash

    # 2. 借出 WETH from Aave v3 pool
    price = get_price()
    weth_amount_to_borrow_in_human = Decimal(usdc_in_wei_to_human(needed_debt_in_wei)) / price
    if (weth_amount_to_borrow_in_human > 0.005): # 小額忽略: < 0.005 WETH
        # 還需要多借出 needed_debt_in_wei 等值的 WETH
        print(f"Need to borrow {weth_amount_to_borrow_in_human} WETH")
        weth_amount_to_borrow_in_wei = eth_in_human_to_wei(weth_amount_to_borrow_in_human)
        amount = int(weth_amount_to_borrow_in_wei)
        interestRateMode = int(change_type(2, 'uint'))
        referralCode = int(change_type(0, uint16))
        borrow_tx = aavev3_pool_contract.functions.borrow(
            weth_address, 
            amount, #weth_amount_to_borrow_in_wei, 
            interestRateMode, #int(2),  # 借款利率模式：1 代表稳定利率，2 代表浮动利率
            referralCode, #int(0),  # referralCode
            wallet_address, 
        ).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
            #'value': eth_amount_to_borrow_in_wei,  # 需要发送的 ETH 数量
            'gas': GAS_LIMIT,  # Gas 设置可能需要调整
            #'gasPrice': web3.toWei('50', 'gwei')
            'type': '0x2', # EIP-1559
        })
        signed_borrow_tx = web3.eth.account.sign_transaction(borrow_tx, wallet_private_key)
        borrow_tx_hash = web3.eth.send_raw_transaction(signed_borrow_tx.rawTransaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(borrow_tx_hash)
        assert tx_receipt.status == 1
        print(f"*** Borrow TX Hash: {borrow_tx_hash.hex()}")
        return borrow_tx_hash
    return 0

def close_CDP():
    print(f"close_CDP()...")
    # 檢查 WETH 夠不夠還, 不夠要買 (加了 0.5% 視為利息及滑價的 buffer)
    price = get_price()
    collateral_in_wei, debt_in_wei = get_aave_pool_status()
    debt_in_human = usdc_in_wei_to_human(debt_in_wei)
    needed_weth_to_repay_in_human = Decimal(debt_in_human) / price * Decimal(1.005)
    # 檢查錢包內是否有足夠的 WETH
    balance_weth_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
    balance_weth_in_human = eth_in_wei_to_human(balance_weth_in_wei)
    adjusted_needed_weth = needed_weth_to_repay_in_human - balance_weth_in_human
    if (adjusted_needed_weth > 0):
        #usdc_to_swap = Decimal(adjusted_needed_weth) * price
        #print(f"Repay neeeds {adjusted_needed_weth} WETH => Swap {usdc_to_swap} USDC")
        #buy_tx_hash = swap_usdc2weth(usdc_to_swap)
        print(f"Repay neeeds {adjusted_needed_weth} WETH => Swap USDC")
        buy_tx_hash = swap_usdc2weth_output(adjusted_needed_weth)
    balance_weth_in_wei = weth_contract.functions.balanceOf(wallet_address).call() - 1
    # 換完後用錢包所有的 WETH 還
    # 1. repay: 还清借出的 WETH
    if (debt_in_wei > 0):
        repay_tx = aavev3_pool_contract.functions.repay(
            weth_address, 
            balance_weth_in_wei, #MAX_UINT_256 -1, # uint(-1) 表示全還
            int(2),  # 利率模式，与借款时相同
            wallet_address
        ).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
            'gas': GAS_LIMIT,  # Gas 设置可能需要调整
            #'gasPrice': web3.toWei('50', 'gwei')
            'type': '0x2', # EIP-1559
        })
        signed_repay_tx = web3.eth.account.sign_transaction(repay_tx, wallet_private_key)
        repay_tx_hash = web3.eth.send_raw_transaction(signed_repay_tx.rawTransaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(repay_tx_hash)
        assert tx_receipt.status == 1
        print(f"*** Repay TX Hash: {repay_tx_hash.hex()}")
        #return repay_tx_hash

    # 2. withdraw: 取回存入的 USDC.e
    if (collateral_in_wei > 0):
        withdraw_tx = aavev3_pool_contract.functions.withdraw(
            usdc_address, 
            MAX_UINT_256, #int(deposit_usdc_in_wei * 1.02), #MAX_UINT_128,  # type(uint).max 表示全部取出
            wallet_address
        ).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_checksum_address),
            #'value': eth_amount_to_borrow_in_wei,  # 需要发送的 ETH 数量
            'gas': GAS_LIMIT,  # Gas 设置可能需要调整
            #'gasPrice': web3.toWei('50', 'gwei')
            'type': '0x2', # EIP-1559
        })
        signed_withdraw_tx = web3.eth.account.sign_transaction(withdraw_tx, wallet_private_key)
        withdraw_tx_hash = web3.eth.send_raw_transaction(signed_withdraw_tx.rawTransaction)
        tx_receipt = web3.eth.wait_for_transaction_receipt(withdraw_tx_hash)
        assert tx_receipt.status == 1
        print(f"*** Withdraw TX Hash: {withdraw_tx_hash.hex()}")
        return withdraw_tx_hash

    return 0

# end of fuctions with tx --------------------------------------------------------------------------

def check_and_adjust_balance(want_weth, at_least_usdc) -> bool: # True:All ok, False:Fail => Stop
    print(f"- check_and_adjust_balance({want_weth}, {at_least_usdc})")
    price = get_price()
    print("- Price: {:.2f}".format(price))
    print("- Want: {:.3f} WETH + {:.3f} USDC".format(want_weth, at_least_usdc))
    balance_weth_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
    balance_weth_in_human = eth_in_wei_to_human(balance_weth_in_wei)
    balance_usdc_in_wei = usdc_contract.functions.balanceOf(wallet_address).call()
    balance_usdc_in_human = usdc_in_wei_to_human(balance_usdc_in_wei)
    print("- Wallet: {:.3f} WETH + {:.3f} USDC".format(balance_weth_in_human, balance_usdc_in_human))
    # 1. 檢查 (錢包裡 WETH 市值 + USDC) 是否大於 (want_weth 市值 + at_least_usdc)
    balance_value = Decimal(balance_weth_in_human) * price + balance_usdc_in_human
    want_value = Decimal(want_weth) * price + at_least_usdc
    if (balance_value < want_value):
        # 錢包資產怎麼換都不夠 => 保持空手 (錢包內全持有 USDC)
        print("Error Message: token value in wallet {:.3f} < want {:.3f}!".format(balance_value, want_value))
        swap_all_weth() # 將 WETH 清空
        return False
    # 2. 執行 WETH 多賣少買
    more_weth = want_weth - balance_weth_in_human
    if (more_weth > 0.005): # WETH 不夠 => 從 USDC 換
        # WETH 不夠, 還需要 more_weth WETH
        #usdc_to_swap = Decimal(more_weth) * price
        #buy_tx_hash = swap_usdc2weth(usdc_to_swap)
        buy_tx_hash = swap_usdc2weth_output(more_weth)
    elif (more_weth < -0.005): # WETH 太多 => 多的換成 USDC
        # WETH 太多 => 賣掉多餘的 -more_weth WETH
        sell_tx_hash = swap_weth2usdc(Decimal(-more_weth))
    return True

def get_range_ticks_and_needed_weth_usdc(total_usdc_amount, min_price, max_price, price, price_tick):
    # 計算 range ticks
    min_tick, max_tick, adjusted_min_price, adjusted_max_price = calculate_range_ticks(price_tick, min_price, max_price)
    print("- Range: ({:.2f} ~ {:.2f})".format(adjusted_min_price, adjusted_max_price))
    # 計算要需要多少 WETH 及 USDC
    needed_weth, needed_usdc = calculate_liquidity_amounts(price, adjusted_min_price, adjusted_max_price, total_usdc_amount)
    print("- Need: {:.3f} WETH + {:.3f} USDC".format(needed_weth, needed_usdc))
    return min_tick, max_tick, needed_weth, needed_usdc

def go_signal(signal): # return 0:success 1:fail
    global ST_position, LP_position, uniswap_newest_LP, Alert_Price
    #ST_position = signal
    print(f"go_signal({signal}), ST_position: {ST_position}, LP_position: {LP_position}")

    # signal == 0 => 清空部位
    if (signal == 0):
        #update_LP_position_and_uniswap_positions()
        if (LP_position > 0):
            print("Current Long => clean Long position...")
            close_LP_all() # to be some WETH and some USDC
        elif (LP_position < 0):
            print("Current Short => clean Short position...")
            if (LP_position != -2): # -2: 已沒有 Uniswap 部位
                close_LP_all() # to be some WETH and some USDC
            close_CDP() # repay WETH and withdraw USDC
        # 將所有 WETH 換成 USDC
        swap_all_weth()
        LP_position = 0


    # R: 設定濾網: 還在 range 內就不動作 --------------------------
    if (LP_position == 1 or LP_position == -1):
        #min_price, max_price = get_1st_alive_LP_range() # 0 means no alive position
        if (Alert_Price):
            price = Alert_Price
        else:
            price = get_price()
        if (uniswap_newest_LP.min_price > 0 and price > uniswap_newest_LP.min_price and price < uniswap_newest_LP.max_price):
            send_message("{:.2f} in ({:.2f} ~ {:.2f})\n  => No action".format(price, uniswap_newest_LP.min_price, uniswap_newest_LP.max_price))
            return 1
    # End: 設定濾網: 還在 range 內就不動作 --------------------------


    # 執行非零 signal
    if (signal > 0): # 0. 已持有多單 => 要調整: 先清空 LP, 再 add_LP
        # 1. 已持有空單 => 清空單
        if (LP_position < 0):
            close_LP_all() # to be some WETH and some USDC
            close_CDP() # repay WETH and withdraw USDC
            LP_position = 0
        elif (LP_position > 0):
            close_LP_all() # to be some WETH and some USDC
            LP_position = 0
        

        # 2. 估計多單 LP 分布: 根據投入 total_usdc_amount USDC, 設定的 range (min, max) 及目前 ETH price, price_tick 來估計所需的 WETH
        price, price_tick = get_price_tick() # refresh price...
        min_price = price - range_stop_loss
        max_price = price + range_take_profit
        min_tick, max_tick, needed_weth, needed_usdc = get_range_ticks_and_needed_weth_usdc(total_usdc_amount, min_price, max_price, price, price_tick)

        # 3. 調整部位: 根據目前錢包擁有的 ETH 及 WETH, 調整到錢包裡 WETH 數量 = needed_weth
        # 3.1 調整 ETH: 錢包只保留 keep_ETH ETH for gas fee, 多餘的先 wrap 成 WETH
        #wrap_all_ETH() # R: 錢包中的 ETH 不要變動???
        # 3.2 調整 WETH: 檢查 WETH 夠不夠, 多賣少買
        if (check_and_adjust_balance(needed_weth, needed_usdc) == False):
            send_message("Stop add_LP since check_and_adjust_balance() failed...")
            return 1

        # 4. 進場: addLP
        #weth_amount_in_wei = eth_in_human_to_wei(needed_weth)
        #usdc_amount_in_wei = usdc_in_human_to_wei(needed_usdc)
        # R: 用所有錢包內的 USDC 當 buffer
        weth_amount_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
        usdc_amount_in_wei = usdc_contract.functions.balanceOf(wallet_address).call()
        add_LP(weth_amount_in_wei, usdc_amount_in_wei, min_tick, max_tick, need_balance_check=False)
        LP_position = 1

    elif (signal < 0): # 0. 已持有空單 => 要調整: 先清空 LP, 再 add_LP
        # 1. 已持有多單 => 清多單
        if (LP_position > 0):
            close_LP_all() # to be some WETH and some USDC
            close_CDP() # repay WETH and withdraw USDC (R:清多單時應該不需要)
            LP_position = 0
        elif (LP_position < 0):
            close_LP_all() # to be some WETH and some USDC
            #close_CDP() # repay WETH and withdraw USDC (R:清多單時應該不需要)
            LP_position = -2 # 調整到一半

        # 2. 估計空單 LP 分布: 根據投入 usdc_amount USDC, 目前 ETH price, 設定的 range (min, max) 來估計所需的 WETH
        price, price_tick = get_price_tick() # refresh price...
        min_price = price - range_take_profit
        max_price = price + range_stop_loss
        min_tick, max_tick, needed_weth, needed_usdc = get_range_ticks_and_needed_weth_usdc(total_usdc_amount, min_price, max_price, price, price_tick)
        
        # 3. 調整部位: 根據目前錢包擁有的 ETH 及 WETH, 調整到錢包裡 WETH 數量 = needed_weth
        # 3.0 從 Aave 借出 WETH: needed_usdc 就是要從 Aave 借出的 "等值 WETH", 例如 needed_usdc = 220, 目前 ETH price = 2200, 就要借出 0.1 WETH
        make_CDP(needed_usdc)
        # 3.1 調整 ETH: 錢包只保留 keep_ETH ETH for gas fee, 多餘的先 wrap 成 WETH
        #wrap_all_ETH() # R: 錢包中的 ETH 可變動也可不變動
        # 3.2 調整 WETH: 檢查 WETH 夠不夠, 多賣少買
        if (check_and_adjust_balance(needed_weth, needed_usdc) == False):
            send_message("Stop add_LP since check_and_adjust_balance() failed...")
            return 1

        # 4. 進場: addLP
        #weth_amount_in_wei = eth_in_human_to_wei(needed_weth)
        #usdc_amount_in_wei = usdc_in_human_to_wei(needed_usdc)
        # R: 用所有錢包內的 USDC 當 buffer
        weth_amount_in_wei = weth_contract.functions.balanceOf(wallet_address).call()
        usdc_amount_in_wei = usdc_contract.functions.balanceOf(wallet_address).call()
        add_LP(weth_amount_in_wei, usdc_amount_in_wei, min_tick, max_tick, need_balance_check=False)
        LP_position = -1
    
    return 0

##-------------------

get_config()

# Connect to Ethereum
web3 = Web3(Web3.HTTPProvider(infura_url))
if not web3.is_connected():
    raise Exception("Unable to connect to Ethereum network.\n")


# checksum address
weth_checksum_address = Web3.to_checksum_address(weth_address) # WETH token0
usdc_checksum_address = Web3.to_checksum_address(usdc_address) # USDC.e token1
uniswapv3_eth_usdc_pool_checksum_address = Web3.to_checksum_address(uniswapv3_eth_usdc_pool_address)
uniswapv3_router2_checksum_address = Web3.to_checksum_address(uniswapv3_router2_address)
uniswapv3_posnft_checksum_address = Web3.to_checksum_address(uniswapv3_posnft_address)
aavev3_pool_checksum_address = Web3.to_checksum_address(aavev3_pool_address)
aavev3_weth_gateway_checksum_address = Web3.to_checksum_address(aavev3_weth_gateway_address)
wallet_checksum_address = Web3.to_checksum_address(wallet_address) # my wallet
# contract
weth_contract = web3.eth.contract(address=weth_checksum_address, abi=weth_abi)
usdc_contract = web3.eth.contract(address=usdc_checksum_address, abi=usdc_abi)
uniswapv3_eth_usdc_pool_contract = web3.eth.contract(address=uniswapv3_eth_usdc_pool_checksum_address, abi=uniswapv3_eth_usdc_pool_abi)
uniswapv3_router2_contract = web3.eth.contract(address=uniswapv3_router2_checksum_address, abi=uniswapv3_router2_abi)
uniswapv3_posnft_contract = web3.eth.contract(address=uniswapv3_posnft_checksum_address, abi=uniswapv3_posnft_abi)
aavev3_pool_contract = web3.eth.contract(address=aavev3_pool_checksum_address, abi=aavev3_pool_abi)
aavev3_weth_gateway_contract = web3.eth.contract(address=aavev3_weth_gateway_checksum_address, abi=aavev3_weth_gateway_abi)

def do_approval(need_check_allowance):
    print(f"do_approval : "+ str(need_check_allowance))
    if (need_check_allowance):
        # for uniswap V3 Router (for swap)
        check_allowance(MAX_USDC_APPROVE_AMOUNT, usdc_contract, uniswapv3_router2_checksum_address)
        check_allowance(MAX_WETH_APPROVE_AMOUNT, weth_contract, uniswapv3_router2_checksum_address)
    
        # for uniswap V3 position NFT (for addLP)
        check_allowance(MAX_USDC_APPROVE_AMOUNT, usdc_contract, uniswapv3_posnft_checksum_address)
        check_allowance(MAX_WETH_APPROVE_AMOUNT, weth_contract, uniswapv3_posnft_checksum_address)
    
        # for aave V3 pool (for lendging/borrowing, for short)
        check_allowance(MAX_USDC_APPROVE_AMOUNT, usdc_contract, aavev3_pool_checksum_address)
        check_allowance(MAX_WETH_APPROVE_AMOUNT, weth_contract, aavev3_pool_checksum_address)

        print(f"do_approval : OK ")