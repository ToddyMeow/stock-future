"""
CTP行情接口
last updated at Aug. 19, 2025
"""

class CTPMdApi:
    """CTP行情接口专业封装"""
    
    def __init__(self, broker_id: str, user_id: str, password: str, md_address: str):
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.md_address = md_address
        
        # 行情数据缓存
        self.tick_data = {}
        self.subscribed_symbols = set()
        
        # 回调函数
        self.on_tick = None
        self.tick_queue = Queue()
        
        # 行情录制
        self.recording = False
        self.tick_records = []
        
    def connect(self):
        """连接行情服务器"""
        print(f"正在连接CTP行情服务器: {self.md_address}")
        # self.api = MdApi.CreateMdApi()
        # self.api.RegisterSpi(self)
        # self.api.RegisterFront(self.md_address)
        # self.api.Init()
        
    def subscribe(self, symbols: list):
        """订阅行情"""
        for symbol in symbols:
            print(f"订阅行情: {symbol}")
            self.subscribed_symbols.add(symbol)
            # self.api.SubscribeMarketData([symbol])
    
    def unsubscribe(self, symbols: list):
        """取消订阅"""
        for symbol in symbols:
            print(f"取消订阅: {symbol}")
            self.subscribed_symbols.discard(symbol)
            # self.api.UnSubscribeMarketData([symbol])
    
    def OnRtnDepthMarketData(self, tick: dict):
        """行情推送回调"""
        symbol = tick['InstrumentID']
        
        # 解析tick数据
        tick_data = {
            'symbol': symbol,
            'last_price': tick['LastPrice'],
            'volume': tick['Volume'],
            'open_interest': tick['OpenInterest'],
            'bid_price': tick['BidPrice1'],
            'bid_volume': tick['BidVolume1'],
            'ask_price': tick['AskPrice1'],
            'ask_volume': tick['AskVolume1'],
            'datetime': f"{tick['TradingDay']} {tick['UpdateTime']}.{tick['UpdateMillisec']}",
            'upper_limit': tick['UpperLimitPrice'],
            'lower_limit': tick['LowerLimitPrice'],
            'pre_close': tick['PreClosePrice'],
            'pre_settlement': tick['PreSettlementPrice'],
        }
        
        # 更新缓存
        self.tick_data[symbol] = tick_data
        
        # 记录行情
        if self.recording:
            self.tick_records.append(tick_data.copy())
        
        # 触发回调
        if self.on_tick:
            self.on_tick(tick_data)
        
        # 加入队列供策略消费
        self.tick_queue.put(tick_data)
    
    def get_last_tick(self, symbol: str) -> dict:
        """获取最新tick"""
        return self.tick_data.get(symbol, None)
    
    def start_recording(self):
        """开始录制行情"""
        self.recording = True
        self.tick_records = []
        print("开始录制行情数据")
    
    def stop_recording(self, filename: str = None):
        """停止录制并保存"""
        self.recording = False
        if filename and self.tick_records:
            with open(filename, 'w') as f:
                json.dump(self.tick_records, f)
            print(f"行情数据已保存到: {filename}")
        return self.tick_records
