"""
CTP交易接口
"""
import threading
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Optional, Callable
from queue import Queue
import json


class CTPTradeApi:
    """CTP交易接口封装"""
    
    def __init__(self, broker_id: str, user_id: str, password: str, 
                 app_id: str, auth_code: str, td_address: str):
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.app_id = app_id
        self.auth_code = auth_code
        self.td_address = td_address
        
        # 会话管理
        self.front_id = None
        self.session_id = None
        self.order_ref = 0
        self.request_id = 0
        
        # 订单管理
        self.orders = {}  # order_ref -> order_data
        self.trades = {}  # trade_id -> trade_data
        self.positions = defaultdict(lambda: {'long': 0, 'short': 0})
        self.accounts = {}
        
        # 线程安全
        self.lock = threading.Lock()
        self.connected = False
        self.login_status = False
        
        # 回调函数注册
        self.callbacks = {
            'on_order': [],
            'on_trade': [],
            'on_error': [],
            'on_position': [],
        }
        
    def connect(self):
        """连接CTP服务器"""
        print(f"正在连接CTP交易服务器: {self.td_address}")
        # 实际实现需要调用CTP API
        # self.api = TraderApi.CreateTraderApi()
        # self.api.RegisterSpi(self)
        # self.api.RegisterFront(self.td_address)
        # self.api.Init()
        self.connected = True
        
    def login(self):
        """登录交易账户"""
        login_req = {
            'BrokerID': self.broker_id,
            'UserID': self.user_id,
            'Password': self.password,
            'AppID': self.app_id,
            'AuthCode': self.auth_code
        }
        print(f"登录请求: 用户{self.user_id}")
        # self.api.ReqUserLogin(login_req, self.get_request_id())
        self.login_status = True
        
    def send_order(self, symbol: str, direction: str, offset: str, 
                   price: float, volume: int, order_type: str = 'LIMIT') -> str:
        """
        发送委托
        
        参数:
            symbol: 合约代码 (如 'rb2405')
            direction: 方向 ('BUY'/'SELL')
            offset: 开平标志 ('OPEN'/'CLOSE'/'CLOSETODAY'/'CLOSEYESTERDAY')
            price: 价格
            volume: 数量
            order_type: 订单类型 ('LIMIT'/'MARKET'/'FAK'/'FOK')
        
        返回:
            order_ref: 订单引用号
        """
        with self.lock:
            self.order_ref += 1
            order_ref = str(self.order_ref)
            
        order_req = {
            'BrokerID': self.broker_id,
            'InvestorID': self.user_id,
            'InstrumentID': symbol,
            'OrderRef': order_ref,
            'UserID': self.user_id,
            'OrderPriceType': self._get_price_type(order_type),
            'Direction': direction,
            'CombOffsetFlag': self._get_offset_flag(offset),
            'CombHedgeFlag': '1',  # 投机
            'LimitPrice': price if order_type == 'LIMIT' else 0,
            'VolumeTotalOriginal': volume,
            'TimeCondition': '3' if order_type in ['FAK', 'FOK'] else '1',  # GFD
            'VolumeCondition': '1',  # 任意数量
            'MinVolume': 1,
            'ContingentCondition': '1',  # 立即
            'ForceCloseReason': '0',  # 非强平
            'IsAutoSuspend': 0,
        }
        
        # 记录订单
        self.orders[order_ref] = {
            'symbol': symbol,
            'direction': direction,
            'offset': offset,
            'price': price,
            'volume': volume,
            'traded_volume': 0,
            'status': 'PENDING',
            'submit_time': datetime.now(),
            'order_type': order_type
        }
        
        print(f"发送订单: {symbol} {direction} {offset} {price}@{volume}")
        # self.api.ReqOrderInsert(order_req, self.get_request_id())
        
        return order_ref
    
    def cancel_order(self, order_ref: str, symbol: str):
        """撤销订单"""
        cancel_req = {
            'BrokerID': self.broker_id,
            'InvestorID': self.user_id,
            'InstrumentID': symbol,
            'OrderRef': order_ref,
            'FrontID': self.front_id,
            'SessionID': self.session_id,
            'ActionFlag': '0',  # 删除
        }
        
        print(f"撤销订单: {order_ref}")
        # self.api.ReqOrderAction(cancel_req, self.get_request_id())
    
    def query_position(self, symbol: str = None):
        """查询持仓"""
        query_req = {
            'BrokerID': self.broker_id,
            'InvestorID': self.user_id,
        }
        if symbol:
            query_req['InstrumentID'] = symbol
            
        # self.api.ReqQryInvestorPosition(query_req, self.get_request_id())
        return self.positions
    
    def query_account(self):
        """查询账户资金"""
        query_req = {
            'BrokerID': self.broker_id,
            'InvestorID': self.user_id,
        }
        # self.api.ReqQryTradingAccount(query_req, self.get_request_id())
        return self.accounts
    
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    # CTP回调函数
    def OnRtnOrder(self, order_data: dict):
        """订单回报"""
        order_ref = order_data['OrderRef']
        if order_ref in self.orders:
            self.orders[order_ref]['status'] = self._parse_order_status(order_data['OrderStatus'])
            self.orders[order_ref]['exchange_id'] = order_data.get('OrderSysID', '')
            
            # 触发回调
            for callback in self.callbacks['on_order']:
                callback(order_data)
    
    def OnRtnTrade(self, trade_data: dict):
        """成交回报"""
        trade_id = trade_data['TradeID']
        order_ref = trade_data['OrderRef']
        
        # 更新订单成交量
        if order_ref in self.orders:
            self.orders[order_ref]['traded_volume'] += trade_data['Volume']
        
        # 记录成交
        self.trades[trade_id] = {
            'symbol': trade_data['InstrumentID'],
            'direction': trade_data['Direction'],
            'offset': trade_data['OffsetFlag'],
            'price': trade_data['Price'],
            'volume': trade_data['Volume'],
            'trade_time': trade_data['TradeTime'],
            'order_ref': order_ref
        }
        
        # 更新持仓
        self._update_position(trade_data)
        
        # 触发回调
        for callback in self.callbacks['on_trade']:
            callback(trade_data)
    
    def OnErrRtnOrderInsert(self, order_data: dict, error_info: dict):
        """订单错误回报"""
        print(f"订单错误: {error_info['ErrorMsg']}")
        for callback in self.callbacks['on_error']:
            callback(error_info)
    
    def _update_position(self, trade_data: dict):
        """更新持仓"""
        symbol = trade_data['InstrumentID']
        volume = trade_data['Volume']
        direction = trade_data['Direction']
        offset = trade_data['OffsetFlag']
        
        with self.lock:
            if direction == 'BUY':
                if offset == 'OPEN':
                    self.positions[symbol]['long'] += volume
                else:  # CLOSE
                    self.positions[symbol]['short'] -= volume
            else:  # SELL
                if offset == 'OPEN':
                    self.positions[symbol]['short'] += volume
                else:  # CLOSE
                    self.positions[symbol]['long'] -= volume
    
    def _get_price_type(self, order_type: str) -> str:
        """转换价格类型"""
        mapping = {
            'LIMIT': '2',
            'MARKET': '1',
            'FAK': '2',
            'FOK': '2',
        }
        return mapping.get(order_type, '2')
    
    def _get_offset_flag(self, offset: str) -> str:
        """转换开平标志"""
        mapping = {
            'OPEN': '0',
            'CLOSE': '1',
            'CLOSETODAY': '3',
            'CLOSEYESTERDAY': '4',
        }
        return mapping.get(offset, '0')
    
    def _parse_order_status(self, status: str) -> str:
        """解析订单状态"""
        mapping = {
            '0': 'ALLTRADED',
            '1': 'PARTTRADED_QUEUEING',
            '2': 'PARTTRADED_NOTQUEUEING',
            '3': 'NOTTRADED_QUEUEING',
            '4': 'NOTTRADED_NOTQUEUEING',
            '5': 'CANCELLED',
        }
        return mapping.get(status, 'UNKNOWN')
    
    def get_request_id(self) -> int:
        """获取请求ID"""
        with self.lock:
            self.request_id += 1
            return self.request_id

