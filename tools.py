from device import Seat, DeviceManager, ErrorCode

# 座椅智能体工具, ErrorCode
from typing import Optional

def adjust_massage_intensity_wrapper(device_manager: DeviceManager):
    def adjust_massage_intensity(device_id: str, massage_intensity: Seat.MassageIntensity) -> str:
        """
        调整按摩椅的按摩强度。
        :param device_id: 按摩椅的设备ID
        :param massage_intensity: 按摩椅的按摩强度
        :return: 调整结果
        """
        device_manager.set_device_state(device_id, 'massage_intensity', massage_intensity)
        return f"按摩椅 {device_id} 的按摩强度已调整为 {massage_intensity}"
    return adjust_massage_intensity

def adjust_ventilation_intensity_wrapper(device_manager: DeviceManager):
    def adjust_ventilation_intensity(device_id: str, ventilation_intensity: Seat.VentilationIntensity) -> str:
        """
        调整座椅通风的强度。
        :param device_id: 座椅的设备ID
        :param massage_intensity: 座椅的通风强度
        :return: 调整结果
        """
        device_manager.set_device_state(device_id, 'ventilation_intensity', ventilation_intensity)
        return f"座椅 {device_id} 的通风强度已调整为 {ventilation_intensity}"
    return adjust_ventilation_intensity

def adjust_massage_mode_wrapper(device_manager: DeviceManager):
    def adjust_massage_mode(device_id: str, massage_mode:str) -> str:
        """
        调整按摩椅的按摩模式。
        :param device_id: 座椅的设备ID
        :param massage_mode: 座椅的按摩模式
        :return: 调整结果
        """
        device_manager.set_device_state(device_id, 'ventilation_mode', massage_mode)
        return f"座椅 {device_id} 的按摩模式已调整为 {massage_mode}"
    return adjust_massage_mode

# 美团智能体工具

def search_shops_by_type_wrapper(device_manager: DeviceManager):
    def search_shops_by_type(shop_type: str) -> str:
        """
        按类型搜索商家。
        :param shop_type: 商家类型，可选值包括：'烧烤'、'快餐'、'面食'、'川菜'、'粤菜'
        :return: 该类型的所有商家信息
        """
        state, code = device_manager.get_device_state("meituan_app", "shop_data")
        if code != ErrorCode.SUCCESS:
            return "获取商家数据失败"
        
        shop_data = state
        shops = shop_data[shop_data['shop_type'] == shop_type][['shop_id', 'shop_name']].drop_duplicates()
        if shops.empty:
            return f"未找到{shop_type}类型的商家"
        
        result = f"\n{shop_type}类商家:\n"
        for _, row in shops.iterrows():
            # 获取该商家的所有菜品价格范围
            shop_prices = shop_data[shop_data['shop_id'] == row['shop_id']]['price']
            min_price = shop_prices.min()
            max_price = shop_prices.max()
            result += f"- {row['shop_name']} (ID: {row['shop_id']}, 价格区间: ￥{min_price}-{max_price})\n"
        return result
    return search_shops_by_type

def get_shop_menu_wrapper(device_manager: DeviceManager):
    def get_shop_menu(shop_id: str) -> str:
        """
        获取指定商家的菜单。
        :param shop_id: 商家ID，例如'S001'
        :return: 该商家的菜单信息
        """
        state, code = device_manager.get_device_state("meituan_app", "shop_data")
        if code != ErrorCode.SUCCESS:
            return "获取商家数据失败"
        
        shop_data = state
        menu = shop_data[shop_data['shop_id'] == shop_id][['shop_name', 'dish_name', 'price']]
        if menu.empty:
            return f"未找到ID为{shop_id}的商家"
        
        shop_name = menu.iloc[0]['shop_name']
        result = f"\n{shop_name}的菜单:\n"
        for _, row in menu.iterrows():
            result += f"- {row['dish_name']} (￥{row['price']})\n"
        return result
    return get_shop_menu

def add_to_cart_wrapper(device_manager: DeviceManager):
    def add_to_cart(dish_name: str) -> str:
        """
        将指定菜品加入购物车。
        :param dish_name: 菜品名称
        :return: 添加结果信息
        """
        # 获取商家数据以检查菜品是否存在
        state, code = device_manager.get_device_state("meituan_app", "shop_data")
        if code != ErrorCode.SUCCESS:
            return "获取商家数据失败"
        
        shop_data = state
        dish_info = shop_data[shop_data['dish_name'] == dish_name]
        if dish_info.empty:
            return f"未找到菜品：{dish_name}"
        
        # 添加到购物车
        meituan_app = device_manager.devices["meituan_app"]
        result = meituan_app.add_to_cart(dish_name)
        
        if result == ErrorCode.SUCCESS:
            shop_name = dish_info.iloc[0]['shop_name']
            price = dish_info.iloc[0]['price']
            return f"成功将 {dish_name} (来自: {shop_name}, 价格: ￥{price}) 加入购物车"
        else:
            return "添加购物车失败"
    return add_to_cart

def add_delivery_location_wrapper(device_manager: DeviceManager):
    def add_delivery_location(order_id: str, location: str) -> str:
        """
        添加订单配送地点。
        :param order_id: 订单ID，例如'order_001'
        :param location: 配送地址
        :return: 添加结果信息
        """
        meituan_app = device_manager.devices["meituan_app"]
        result = meituan_app.add_delivery_location(order_id, location)
        
        if result == ErrorCode.SUCCESS:
            return f"成功添加配送地点\n订单号: {order_id}\n地址: {location}"
        else:
            return "添加配送地点失败"
    return add_delivery_location

def process_payment_wrapper(device_manager: DeviceManager):
    def process_payment(order_id: str, payment_method: str = "微信支付") -> str:
        """
        订单支付付款。
        :param order_id: 订单ID
        :param payment_method: 支付方式，默认为"微信支付"
        :return: 支付结果信息
        """
        # 获取购物车信息以计算总价
        state, code = device_manager.get_device_state("meituan_app", "shopping_cart")
        if code != ErrorCode.SUCCESS:
            return "获取购物车数据失败"
        
        cart = state
        if not cart:
            return "购物车为空，无法支付"
            
        # 获取商家数据以查询价格
        state, code = device_manager.get_device_state("meituan_app", "shop_data")
        if code != ErrorCode.SUCCESS:
            return "获取商家数据失败"
            
        shop_data = state
        total_price = 0
        order_details = []
        
        for item in cart:
            dish_info = shop_data[shop_data['dish_name'] == item['dish_name']].iloc[0]
            price = dish_info['price']
            total_price += price
            order_details.append(f"- {item['dish_name']} (￥{price})")
        
        result = f"订单 {order_id} 支付详情:\n"
        result += "\n".join(order_details)
        result += f"\n总价: ￥{total_price}"
        result += f"\n支付方式: {payment_method}"
        result += "\n支付状态: 成功"
        
        return result
    return process_payment

# FunctionTool(adjust_massage_intensity, description="调整座椅的按摩强度，需要提供座椅的id和按摩强度")

def search_music(music_name: str) -> str:
    """
    搜索音乐。
    :param music_name: 音乐名称
    :return: 搜索结果
    """
    return f"[搜索到音乐 {music_name}]"

def search_video(video_name: str) -> str:
    """
    搜索视频。
    :param video_name: 视频名称
    :return: 搜索结果
    """
    return f"[搜索到视频 {video_name}]"

def play_music_wrapper(device_manager: DeviceManager):
    def play_music(device_id: str=None,
                   intent: Optional[str]=['播放', '暂停', '继续', '关闭'],
                   music_name: str=None,
                   ) -> str:
        """
        控制音乐播放器，负责 播放/暂停/关闭。
        :param device_id: 音响的设备ID
        :param intent: 用户的意图，共有三种：播放，暂停，继续，关闭
        :param music_name: 音乐名称（如果意图不是播放音乐，则不用设置）
        :return: 播放结果
        """
        if intent == '播放':
            print(search_music(music_name))
            device_manager.set_device_state(device_id, 'play', True)
            device_manager.set_device_state(device_id, 'current_music', music_name)
            return f"[音乐 {music_name} 已开始播放]"
        elif intent == '暂停':
            current_music = device_manager.get_device_state(device_id, 'current_music')
            if not current_music:
                return f"[当前没有音乐在播放]"
            else:
                device_manager.set_device_state(device_id, 'pasue', True)
                return f"[音乐{music_name}已暂停]"
        elif intent == '继续':
            current_music = device_manager.get_device_state(device_id, 'current_music')
            if not current_music:
                return f"[当前没有音乐在播放]"
            else:
                device_manager.set_device_state(device_id, 'pasue', False)
                return f"[音乐{music_name}已继续播放]"
        elif intent == '关闭':
            device_manager.set_device_state(device_id, 'play', False)
            device_manager.set_device_state(device_id, 'current_music', None)
            return f"[音乐已关闭]"
        else:
            return f"[不支持的操作]"
        
    return play_music


def play_video_wrapper(device_manager: DeviceManager):
    def play_video(device_id: str=None,
                   intent: Optional[str]=['播放', '暂停', '继续', '关闭'],
                   video_name: str=None,
                   ) -> str:
        """
        控制视频播放器，负责 播放/暂停/关闭。
        :param device_id: 视频播放器的设备ID
        :param intent: 用户的意图，共有三种：播放，暂停，继续，关闭
        :param video_name: 视频名称（如果意图不是播放视频，则不用设置）
        :return: 播放结果
        """
        if intent == '播放':
            print(search_video(video_name))
            device_manager.set_device_state(device_id, 'play', True)
            device_manager.set_device_state(device_id, 'current_video', video_name)
            return f"[视频 {video_name} 已开始播放]"
        elif intent == '暂停':
            current_video = device_manager.get_device_state(device_id, 'current_video')
            if not current_video:
                return f"[当前没有视频在播放]"
            else:
                device_manager.set_device_state(device_id, 'pasue', True)
                return f"[视频{video_name}已暂停]"
        elif intent == '继续':
            current_video = device_manager.get_device_state(device_id, 'current_video')
            if not current_video:
                return f"[当前没有视频在播放]"
            else:
                device_manager.set_device_state(device_id, 'pasue', False)
                return f"[视频{video_name}已继续播放]"
        elif intent == '关闭':
            device_manager.set_device_state(device_id, 'play', False)
            device_manager.set_device_state(device_id, 'current_video', None)
            return f"[视频已关闭]"
        else:
            return f"[不支持的操作]"
    
    return play_video

def get_Cabin_Temperature_wrapper(device_manager: DeviceManager):
    def get_Cabin_Temperature(device_id: str):
        """
        获取车内或车外温度
        :param device_id: 温度的设备ID
        :return：车内/车外温度
        """
        state, code = device_manager.get_device_state(device_id, "temperature")
        if code == ErrorCode.SUCCESS:
            return f"{state}℃"
        return "温度查询失败"
    return get_Cabin_Temperature

def get_Tire_Status_wrapper(device_manager: DeviceManager):
    def get_Tire_Status(device_id:str, entitySub: str) -> str:
        """
        获取轮胎状态
        :param device_id: 轮胎的设备ID
        :param entitySub: 轮胎的状态种类 胎压/胎温
        :return: 轮胎状态
        """
        status_map = {
            "胎压": {
                0: "正常", 1: "过高", 2: "过低",
                3: "快速漏气", 4: "传感器丢失", 5: "电量低"
            },
            "胎温": {
                0: "正常", 1: "温度过高"
            }
        }
         
        feature = "tire_pressure_status" if entitySub == "胎压" else "tire_temperature_status"
        state, code = device_manager.get_device_state(device_id, feature)
        
        if code == ErrorCode.SUCCESS:
            return status_map[entitySub][state]
        return "状态查询失败"
    return get_Tire_Status

def get_CurrentTime_wrapper(device_manager: DeviceManager):
    def get_CurrentTime(device_id: str):
        """
        获取当前时间
        :param device_id: 时钟设备ID
        :return: 当前时间
        """
        state, code = device_manager.get_device_state(device_id, "current_time")
        return state if code == ErrorCode.SUCCESS else "时间查询失败"
    return get_CurrentTime

def get_SOC_wrapper(device_manager: DeviceManager):
    def get_SOC(device_id: str):
        """
        获取电池电量
        :param device_id: 电池设备ID
        :return: 当前电池电量
        """
        state, code = device_manager.get_device_state(device_id, "soc")
        return f"{state}%" if code == ErrorCode.SUCCESS else "电量查询失败"
    return get_SOC

def get_GPS_Location_wrapper(device_manager: DeviceManager):
    def get_GPS_Location(device_id: str):
        """
        获取当前位置坐标
        :param device_id: GPS定位设备ID
        :return: 当前位置坐标，（经度，纬度）
        """
        longitude, code = device_manager.get_device_state(device_id, "longitute") 
        latitude, code = device_manager.get_device_state(device_id, "latitude") 
        
        return f"经度为东经{longitude}度，纬度为北纬{latitude}度" if code == ErrorCode.SUCCESS else "位置坐标查询失败"
    return get_GPS_Location

def get_Passager_wrapper(device_manager: DeviceManager):
    def get_Passager(device_id: str):
        """
        获取当前摄像头下的用户所属ID
        :param device_id: 摄像头设备ID
        :return: 当前摄像头下的乘客ID
        """
        occ,_ = device_manager.get_device_state(device_id, "occupancy")
        if occ == True:
            state, code = device_manager.get_device_state(device_id, "face_id")
            return f"当前乘客为用户{state}" if code == ErrorCode.SUCCESS else "乘客ID查询失败"
        else:
            return "当前位置无乘客"
    return get_Passager

def get_Voice_wrapper(device_manager: DeviceManager):
    def get_Voice(device_id: str):
        """
        获取当前说话人的声音ID
        :param device_id: 声音设备ID
        :return: 当前说话人ID
        """
        state, code = device_manager.get_device_state(device_id, "voice_id")
        return f"当前说话用户为用户{state}" if code == ErrorCode.SUCCESS else "乘客声音ID查询失败"
    return get_Voice

def search_path(start: str, end : str) -> str:
    return f"{start} -> {end}."

def get_path_wrapper(device_manager: DeviceManager):
    def get_path(device_id: str = None,
                 start: str = None,
                 end: str = None):
        """
        控制视频播放器，负责 播放/暂停/关闭。
        :param device_id: 导航设备ID
        :param intent: 意图 : 导航 or 推荐
        :param start: 始发地
        :param end: 目的地
        :return: 播放结果
        """
        path = search_path(start,end)
        device_manager.set_device_state(device_id, 'start', start)
        device_manager.set_device_state(device_id, 'end', end)
        return search_path(start,end)
    return get_path

