from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Type, List, Tuple
import json
import pandas as pd
from datetime import datetime

class ErrorCode(Enum):
    SUCCESS = 0
    DEVICE_NOT_FOUND = 1
    FEATURE_NOT_FOUND = 2
    INVALID_VALUE = 3
    DUPLICATE_DEVICE_ID = 4

class DeviceFeature:
    """设备功能特征描述类"""
    def __init__(self, name: str, value_type: Type, valid_values: Any = None, description:str=""):
        self.name = name
        self.value_type = value_type
        self.valid_values = valid_values
        self.description = description

    def validate(self, value: Any) -> bool:
        """验证输入值的合法性"""
        if not isinstance(value, self.value_type):
            return False
        if self.valid_values is not None:
            if isinstance(self.valid_values, range):
                return value in self.valid_values
            return value in self.valid_values
        return True

class BaseDevice(ABC):
    """设备基类"""
    def __init__(self, device_id: str):
        self.device_id = device_id
        self._state: Dict[str, Any] = {}
        self._features: Dict[str, DeviceFeature] = {}
        self._initialize_features()
        
    @abstractmethod
    def _initialize_features(self):
        """初始化设备功能特征（由子类实现）"""
        pass

    def get_state(self, feature: str) -> Tuple[Any, ErrorCode]:
        """获取指定功能状态"""
        if feature not in self._features:
            return None, ErrorCode.FEATURE_NOT_FOUND
        return self._state.get(feature), ErrorCode.SUCCESS

    def set_state(self, feature: str, value: Any) -> ErrorCode:
        """设置功能状态（带合法性检查）"""
        if feature not in self._features:
            return ErrorCode.FEATURE_NOT_FOUND
        
        feature_desc = self._features[feature]
        if not feature_desc.validate(value):
            return ErrorCode.INVALID_VALUE
        
        self._state[feature] = value
        self._post_state_change(feature, value)
        return ErrorCode.SUCCESS

    def _post_state_change(self, feature: str, value: Any) -> None:
        """状态变更后处理钩子方法"""
        pass

    def __str__(self) -> str:
            """设备类型名（由子类实现）"""
            pass
    
    def get_device_features_info(self, feature_name: str|None=None, format_type: str = 'natural'):
        template = "当前设备类型：\"{}\"，可控属性{}的具体功能是{}"
        if feature_name is not None:
            assert feature_name in self._features, f"请求的feature: {feature_name} 不在注册的device feature中"

        if format_type == 'natural':
            if feature_name is None:
                info = []
                for k, v in self._features.items():
                    info.append(template.format(str(self), k, v.description))
                return '\n'.join(info)
            else:
                return template.format(str(self),  feature_name, self._features[feature_name].description)
        elif format_type == 'json':
            if feature_name is None:
                info = {}
                for k, v in self._features.items():
                    info[k] = v.description
                return json.dumps(info)
            else:
                return json.dumps({feature_name: self._features[feature_name].description})


    @property
    def state(self) -> Dict[str, Any]:
        """返回完整状态快照"""
        return self._state.copy()

class Seat(BaseDevice):
    class Position(Enum):
        DRIVER = "driver"
        PASSENGER = "passenger"
        REAR_LEFT = "rear_left"
        REAR_RIGHT = "rear_right"

    class MassageIntensity(Enum):
        OFF = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    class VentilationIntensity(Enum):
        OFF = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    def __init__(self, position: Position):
        super().__init__(f"seat_{position.value}")
        self.position = position
    
    def __str__(self) -> str:
        return '座椅，Seat'
    
    def _initialize_features(self):
        self._features.update({
            'occupancy': DeviceFeature('occupancy', bool, description="当前座位是否有人"),
            'massage_mode': DeviceFeature('massage_mode', str, {'wave', 'pulse'}, description="按摩的模式，有两个可用值，其中`wave`代表波浪按摩，`pulse`代表脉冲按摩"),
            'massage_intensity': DeviceFeature('massage_intensity', 
                                                self.MassageIntensity,
                                                self.MassageIntensity, description="按摩的强度"),
            'ventilation_intensity': DeviceFeature('ventilation_intensity', 
                                                self.VentilationIntensity,
                                                self.VentilationIntensity, description="座椅通风的强度")
        })
        self._state = {
            'occupancy': False,
            'massage_mode': 'wave',
            'massage_intensity': self.MassageIntensity.OFF,
            'ventilation_intensity':self.VentilationIntensity.OFF
        }

class AirConditioner(BaseDevice):
    class FanSpeed(Enum):
        AUTO = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    def __init__(self, zone: str):
        super().__init__(f"ac_{zone}")

    def _initialize_features(self):
        self._features.update({
            'power': DeviceFeature('power', bool),
            'temperature': DeviceFeature('temperature', int, range(16, 31)),
            'fan_speed': DeviceFeature('fan_speed', self.FanSpeed, self.FanSpeed)
        })
        self._state = {
            'power': False,
            'temperature': 26,
            'fan_speed': self.FanSpeed.AUTO
        }

class MeituanApp(BaseDevice):
    def __init__(self):
        # 首先初始化本地数据
        self._initialize_shop_data()
        # 然后调用父类初始化方法
        super().__init__("meituan_app")
    
    def __str__(self) -> str:
        return '美团应用，MeituanApp'
    
    def _initialize_shop_data(self):
        """初始化商家数据"""
        # 示例数据，后续用可用csv表代替
        data = {
            'shop_id': [
                'S001', 'S001', 'S001',                    
                'S002', 'S002',                            
                'S003', 'S003', 'S003',                    
                'S004', 'S004', 'S004',                    
                'S005',                                    
                'S006', 'S006',                           
                'S007', 'S007', 'S007',                   
                'S008', 'S008',                            
                'S009', 'S009', 'S009'                     
            ],
            'shop_name': [
                '老张烤肉', '老张烤肉', '老张烤肉',          
                '小李炒饭', '小李炒饭',
                '大王面馆', '大王面馆', '大王面馆',
                '京都烤肉', '京都烤肉', '京都烤肉',
                '阿香米线',
                '川味小馆', '川味小馆',
                '粤式茶餐厅', '粤式茶餐厅', '粤式茶餐厅',
                '沙县小吃', '沙县小吃',
                '胖哥烤肉', '胖哥烤肉', '胖哥烤肉'
            ],
            'shop_type': [
                '烧烤', '烧烤', '烧烤',                      
                '快餐', '快餐',
                '面食', '面食', '面食',
                '烧烤', '烧烤', '烧烤',                      
                '面食',
                '川菜', '川菜',
                '粤菜', '粤菜', '粤菜',
                '快餐', '快餐',
                '烧烤', '烧烤', '烧烤'
            ],
            'dish_name': [
                '烤羊肉', '烤牛肉', '烤鸡翅',
                '蛋炒饭', '扬州炒饭',
                '牛肉面', '阳春面', '炸酱面',
                '烤五花肉', '烤牛肉', '烤生蚝',
                '肉末米线',
                '麻婆豆腐', '回锅肉',
                '叉烧饭', '白切鸡', '虾饺',
                '三鲜炒面', '扬州炒饭',
                '烤羊肉', '烤五花肉', '烤牛肉'
            ],
            'price': [  
                68, 88, 32,           
                16, 20,               
                28, 16, 22,          
                98, 108, 88,         
                22,                   
                38, 48,             
                32, 48, 28,         
                15, 20,              
                58, 78, 88          
            ]
        }
        self._shop_data = pd.DataFrame(data)
    
    def _initialize_features(self):
        """初始化设备功能特征"""
        self._features.update({
            'shop_data': DeviceFeature(
                'shop_data', 
                pd.DataFrame, 
                description="商家数据表，包含商家ID、名称、类型、菜品名和价格"
            ),
            'shopping_cart': DeviceFeature(
                'shopping_cart',
                list,
                description="购物车，存储菜品名及对应商家ID"
            ),
            'delivery_locations': DeviceFeature(
                'delivery_locations',
                dict,
                description="配送地点信息，存储订单ID及其配送地点"
            )
        })
        
        # 设置初始状态
        self._state = {
            'shop_data': self._shop_data,
            'shopping_cart': [],  # 列表中存储字典 [{'dish_name': '烤羊肉', 'shop_id': 'S001'}, ...]
            'delivery_locations': {}  # {'order_001': '北京市海淀区xx街xx号', ...}
        }
    
    def add_to_cart(self, dish_name: str) -> ErrorCode:
        """添加菜品到购物车"""
        # 查找菜品对应的商家ID
        dish_info = self._shop_data[self._shop_data['dish_name'] == dish_name]
        if dish_info.empty:
            return ErrorCode.INVALID_VALUE
        
        shop_id = dish_info.iloc[0]['shop_id']
        self._state['shopping_cart'].append({
            'dish_name': dish_name,
            'shop_id': shop_id
        })
        return ErrorCode.SUCCESS
    
    def add_delivery_location(self, order_id: str, location: str) -> ErrorCode:
        """添加配送地点"""
        if not order_id or not location:
            return ErrorCode.INVALID_VALUE
        
        self._state['delivery_locations'][order_id] = location
        return ErrorCode.SUCCESS
    
    def clear_cart(self) -> None:
        """清空购物车"""
        self._state['shopping_cart'] = []


class MusicPlayer(BaseDevice):
    
    def __init__(self):
        super().__init__(device_id="music_player")
    
    def __str__(self) -> str:
        return '音乐播放器，MusicPlayer'
    
    def _initialize_features(self):
        self._features.update({
            'play': DeviceFeature('play', bool, description="播放音乐"),
            'pause': DeviceFeature('pause', bool, description="暂停音乐"),
            'volume': DeviceFeature('volume', int, range(0, 101), description="音量大小"),
            "current_music": DeviceFeature('current_music', str, description="当前播放的音乐")
        })
        self._state = {
            'play': False,
            'pause': False,
            'volume': 50,
            "current_music": None,
        }


class VideoPlayer(BaseDevice):
        
        def __init__(self):
            super().__init__(device_id="video_player")
        
        def __str__(self) -> str:
            return '视频播放器，VideoPlayer'
        
        def _initialize_features(self):
            self._features.update({
                'play': DeviceFeature('play', bool, description="播放视频"),
                'pause': DeviceFeature('pause', bool, description="暂停视频"),
                'volume': DeviceFeature('volume', int, range(0, 101), description="音量大小"),
                "current_video": DeviceFeature('current_video', str, description="当前播放的视频")
            })
            self._state = {
                'play': False,
                'pause': False,
                'volume': 50,
                "current_video": None
            }

class Navigator(BaseDevice):

    class FavoriteLoc:  
        def __init__(self):
            self.home: List[str] = []      
            self.company: List[str] = []    
            self.others: List[Dict[str, str]] = [] 

    def __init__(self):
        super().__init__(device_id='navigator')

    def __str__(self) -> str:
        return '出行系统'

    def _initialize_features(self):
        self._features.update({
            'end': DeviceFeature('end', str, description="目的地"),
            'start': DeviceFeature('start', str, description="始发地"),
            'path': DeviceFeature('path', List[List[str]], description="推荐的几个路径"),
            'favorite' : DeviceFeature('favorite',self.FavoriteLoc, description="收藏的地点：(home,company,others)")
        })
        self._state = {
            'end': None,
            'start': None,
            'path': None,
            'favorite' : None
        }
        
class Temperature_device(BaseDevice):
    """车内外温度传感器设备"""
    class Position(Enum):
        INSIDE = "inside"
        OUTSIDE = "outside"

    def __str__(self) -> str:
        return '温度设备，Temperature'
        
    def __init__(self, position:Position):
        super().__init__(f"temperature_{position.value}")
        self.position = position

    def _initialize_features(self):
        self._features.update({
            'temperature': DeviceFeature(
                name="temperature",
                value_type=int,
                valid_values=range(-20, 50),
                description="车内或车外温度状态")
        })
        self._state = {
            'temperature': 26
        }

class Tire(BaseDevice):
    """轮胎设备"""
    class Position(Enum):
        LEFT_FRONT = "left_front"
        RIGHT_FRONT = "right_front"
        LEFT_REAR = "left_rear"
        RIGHT_REAR = "right_rear"

    def __str__(self) -> str:
        return '轮胎设备，Tire'
    
    def __init__(self, position: Position):
        super().__init__(f"tire_{position.value}")
        self.position = position

    def _initialize_features(self):
        self._features.update({
            "tire_pressure_status": DeviceFeature(
                name="tire_pressure_status",
                value_type=int,
                valid_values=range(6),
                description="胎压状态"
            ),
            "tire_temperature_status": DeviceFeature(
                name="tire_temperature_status",
                value_type=int,
                valid_values=[0, 1],
                description="胎温状态"
            )
        })
        self._state = {
            'tire_pressure_status': 0,
            'tire_temperature_status': 0
        }

class Battery(BaseDevice):
    """电池设备"""
    def __init__(self):
        super().__init__("battery")

    def __str__(self) -> str:
            return '电池，battery'

    def _initialize_features(self):
        self._features.update({
            "soc": DeviceFeature(
                name="soc",
                value_type=int,
                valid_values=range(0, 101),
                description="剩余电量"
            )
        })
        self._state = {
            "soc": 80,
        }

class Timeclock(BaseDevice):
    """时间设备"""
    def __init__(self):
        super().__init__("system_time")
        self._initialize_features()
        self._update_time()

    def __str__(self) -> str:
        return '时钟，Timeclock'
    def _initialize_features(self):
        self._features.update({
            "current_time": DeviceFeature(
                name="current_time",
                value_type=str,
                description="当前时间"
            )
        })

    def _update_time(self):
        self.set_state("current_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_state(self, feature: str) -> tuple[any, ErrorCode]:
        self._update_time()  # 每次查询时更新时间
        return super().get_state(feature)

class GPSDevice(BaseDevice):
    """GPS设备"""
    def __init__(self):
        super().__init__("GPS")
        self._initialize_features()

    def __str__(self) -> str:
        return 'gps定位系统，GPSDevice'

    def _initialize_features(self):
        self._features.update({
            "longitude":DeviceFeature(
                name = "longitude",
                value_type = float,
                description="当前车所在位置的经度"
            ),
            "latitude":DeviceFeature(
                name= "latitude",
                value_type = float,
                description="当前车所在位置的纬度" 
            )
        })
        self._state = {
            "longitude": 116.35,
            "latitude": 39.96
        }

class Camera(BaseDevice):
    """车载摄像头"""
    class Position(Enum):
        DRIVER = "driver"
        PASSENGER = "passenger"
        REAR_LEFT = "rear_left"
        REAR_RIGHT = "rear_right"
    def __init__(self, position: Position):
        super().__init__(f"camera_{position.value}")
        self._initialize_features()
        self.position = position
    def __str__(self) -> str:
        return '车内摄像头，Camera'
    def _initialize_features(self):
        self._features.update({
            "occupancy":DeviceFeature(
                name="occupancy",
                value_type=bool,
                description="当前车内摄像头下是否有人"
            ),
            "face_id": DeviceFeature(
                name="face_id",
                value_type=int,
                valid_values=range(0,101),
                description="当前车内摄像头下用户的面容ID，如果当前摄像头下没人则将ID设置为0（有人时将为1到100）"
            )
        }    
        )
        self._state = {
            "occupancy": True,
            "face_id": 1,
        }

class Voice(BaseDevice):
    """声音设备"""
    def __init__(self):
        super().__init__("Voice_device")
        self._initialize_features()
    def __str__(self) -> str:
        return '声音设备， Voice'
    def _initialize_features(self):
        self._features.update({
            "voice_id":DeviceFeature(
                name="voice_id",
                value_type=int,
                valid_values=range(0,101),
                description="当前说话用户所属ID"
            )
        })
        self._state = {
            "voice_id": 0
        }

class DeviceManager:
    """增强型设备管理器"""
    def __init__(self):
        self.devices: Dict[str, BaseDevice] = {}
        self.type_index: Dict[Type[BaseDevice], List[str]] = {}
    
    def register_device(self, device: BaseDevice) -> ErrorCode:
        """设备注册接口"""
        if device.device_id in self.devices:
            return ErrorCode.DUPLICATE_DEVICE_ID
        
        self.devices[device.device_id] = device
        device_type = type(device)
        if device_type not in self.type_index:
            self.type_index[device_type] = []
        self.type_index[device_type].append(device.device_id)
        return ErrorCode.SUCCESS

    def set_device_state(self, device_id: str, feature: str, value: Any) -> ErrorCode:
        """单设备状态设置"""
        if device_id not in self.devices:
            return ErrorCode.DEVICE_NOT_FOUND
        return self.devices[device_id].set_state(feature, value)

    def batch_set_state(self, device_type: Type[BaseDevice], feature: str, value: Any) -> Dict[str, ErrorCode]:
        """批量状态设置"""
        results = {}
        for dev_id in self.type_index.get(device_type, []):
            device = self.devices[dev_id]
            results[dev_id] = device.set_state(feature, value)
        return results

    def get_device_state(self, device_id: str, feature: str = None) -> Tuple[Any, ErrorCode]:
        """状态查询接口"""
        if device_id not in self.devices:
            return None, ErrorCode.DEVICE_NOT_FOUND
        device = self.devices[device_id]
        return device.get_state(feature) if feature else (device.state, ErrorCode.SUCCESS)
    
    
    def collect_environment_info(self, format_type: str = 'json', device_type: Type[BaseDevice] | List[Type[BaseDevice]] | None = None):
        """
        收集设备的id和各自的状态作为环境信息，并根据指定格式返回。
        :param format_type: 输出格式，支持 'json' 或 'natural_language'
        :param device_type: 可接受的设备类型或类型列表，None表示所有类型
        :return: 格式化后的环境信息
        """
        
        # 统一处理device_type为列表形式
        if device_type is not None:
            device_types = [device_type] if not isinstance(device_type, list) else device_type
        else:
            device_types = None
        
        # 收集所有目标设备的状态
        target_devices_states = {}
        if device_types is not None:
            for t in device_types:
                states = self.batch_get_state(t)
                target_devices_states.update(states)
        else:
            for device_id, device in self.devices.items():
                target_devices_states[device_id] = device.state
        
        # 按类型组织设备信息
        target_devices_info = []
        if device_types is not None:
            for t in device_types:
                typed_device_ids = self.type_index.get(t, [])
                if not typed_device_ids:
                    continue  # 跳过无设备的类型
                features_description = self.devices[typed_device_ids[0]].get_device_features_info()
                typed_devices_states = {dev_id: target_devices_states[dev_id] for dev_id in typed_device_ids}
                target_devices_info.append({
                    'device_type': t.__name__,  # 使用类名作为类型标识
                    'features_description': features_description,
                    'devices_states': typed_devices_states
                })
        else:
            for t, typed_device_ids in self.type_index.items():
                features_description = self.devices[typed_device_ids[0]].get_device_features_info()
                typed_devices_states = {dev_id: target_devices_states[dev_id] for dev_id in typed_device_ids}
                target_devices_info.append({
                    'device_type': t.__name__,
                    'features_description': features_description,
                    'devices_states': typed_devices_states
                })

        # 所有枚举类型返回其值，便于json.dumps
        def custom_serializer(obj):
            if isinstance(obj, Enum):
                return obj.value  
            elif isinstance(obj, pd.core.frame.DataFrame):
                return str(obj)  
            raise TypeError(f"无法序列化 {type(obj)}")
        # 格式化输出
        if format_type == 'json':
            return json.dumps(target_devices_info, ensure_ascii=False, indent=2,default=custom_serializer)
        elif format_type == 'natural_language':
            info = []
            for typed_info in target_devices_info:
                info.append(f"设备类型：{typed_info['device_type']}")
                info.append(f"功能描述：{typed_info['features_description']}")
                for dev_id, state in typed_info['devices_states'].items():
                    info.append(f"设备 {dev_id} 的状态：{state}")
                info.append("")  # 添加空行分隔不同设备类型
            return '\n'.join(info).strip()
        else:
            raise ValueError("不支持的格式类型，仅支持 'json' 或 'natural_language'")

    def get_type_devices_ids(self, device_type:BaseDevice):
        """
        获取管理中的指定类型的device的对应id列表
        """
        deivce_ids = self.type_index[device_type]
        return deivce_ids

    def batch_get_state(self, device_type: BaseDevice, feature: str|None=None) -> dict[str, ErrorCode]:
        """批量状态设置"""
        results = {}
        if feature is not None:
            for dev_id in self.type_index.get(device_type, []):
                device = self.devices[dev_id]
                results[dev_id] = device.get_state(feature)
        else:
            for dev_id in self.type_index.get(device_type, []):
                device = self.devices[dev_id]
                results[dev_id] = device.state
        return results

def test_meituan_app():
    # 创建设备管理器
    manager = DeviceManager()
    
    # 创建并注册美团App设备
    meituan_app = MeituanApp()
    result = manager.register_device(meituan_app)
    print(f"注册美团App: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 1. 按类型搜索商家
    print("\n=== 按类型搜索商家 ===")
    state, code = manager.get_device_state("meituan_app", "shop_data")
    if code == ErrorCode.SUCCESS:
        shop_data = state
        # 搜索烧烤类商家
        bbq_shops = shop_data[shop_data['shop_type'] == '烧烤'][['shop_id', 'shop_name']].drop_duplicates()
        print("\n烧烤类商家:")
        print(bbq_shops)
        
        # 搜索面食类商家
        noodle_shops = shop_data[shop_data['shop_type'] == '面食'][['shop_id', 'shop_name']].drop_duplicates()
        print("\n面食类商家:")
        print(noodle_shops)
        
        # 搜索快餐类商家
        fast_food_shops = shop_data[shop_data['shop_type'] == '快餐'][['shop_id', 'shop_name']].drop_duplicates()
        print("\n快餐类商家:")
        print(fast_food_shops)
    
    # 2. 获取指定商家的菜单（包含价格）
    print("\n=== 获取指定商家菜单 ===")
    if code == ErrorCode.SUCCESS:
        # 获取老张烤肉的菜单
        shop_menu = shop_data[shop_data['shop_id'] == 'S001'][['shop_name', 'dish_name', 'price']]
        print("\n老张烤肉的菜单:")
        for _, row in shop_menu.iterrows():
            print(f"- {row['dish_name']} (￥{row['price']})")
        
        # 获取京都烤肉的菜单（高档烧烤）
        shop_menu = shop_data[shop_data['shop_id'] == 'S004'][['shop_name', 'dish_name', 'price']]
        print("\n京都烤肉的菜单:")
        for _, row in shop_menu.iterrows():
            print(f"- {row['dish_name']} (￥{row['price']})")
        
        # 获取沙县小吃的菜单（平价快餐）
        shop_menu = shop_data[shop_data['shop_id'] == 'S008'][['shop_name', 'dish_name', 'price']]
        print("\n沙县小吃的菜单:")
        for _, row in shop_menu.iterrows():
            print(f"- {row['dish_name']} (￥{row['price']})")
    
    # 3. 测试购物车功能
    print("\n=== 测试购物车功能 ===")
    # 添加不同价位的菜品
    test_dishes = [
        '烤羊肉',    # 老张烤肉 68元
        '牛肉面',    # 大王面馆 28元
        '蛋炒饭',    # 小李炒饭 16元
        '烤生蚝'     # 京都烤肉 88元
    ]
    
    print("\n添加多个菜品到购物车:")
    for dish in test_dishes:
        result = meituan_app.add_to_cart(dish)
        dish_info = shop_data[shop_data['dish_name'] == dish].iloc[0]
        print(f"添加 {dish} (￥{dish_info['price']}) 来自 {dish_info['shop_name']}: "
              f"{'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 查看购物车状态
    state, code = manager.get_device_state("meituan_app", "shopping_cart")
    if code == ErrorCode.SUCCESS:
        print("\n当前购物车:")
        cart = state
        total = 0
        # 显示购物车中的菜品、商家和价格
        for item in cart:
            dish_info = shop_data[shop_data['dish_name'] == item['dish_name']].iloc[0]
            price = dish_info['price']
            total += price
            print(f"- {item['dish_name']} (来自: {dish_info['shop_name']}, 价格: ￥{price})")
        print(f"购物车总价: ￥{total}")
    
    # 4. 测试配送地点功能
    print("\n=== 测试配送地点功能 ===")
    # 添加多个配送地点
    delivery_locations = {
        'order_001': '北京市海淀区中关村大街1号',
        'order_002': '北京市朝阳区建国路2号',
        'order_003': '北京市西城区西长安街3号'
    }
    
    for order_id, location in delivery_locations.items():
        result = meituan_app.add_delivery_location(order_id, location)
        print(f"\n添加配送地点 {order_id}: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
        print(f"地址: {location}")
    
    # 查看所有配送地点
    state, code = manager.get_device_state("meituan_app", "delivery_locations")
    if code == ErrorCode.SUCCESS:
        print("\n所有配送地点信息:")
        for order_id, location in state.items():
            print(f"订单 {order_id}: {location}")
    
    # 5. 测试支付功能
    print("\n=== 测试支付功能 ===")
    # 使用不同的支付方式
    payment_methods = ["微信支付", "支付宝", "银行卡"]
    for i, method in enumerate(payment_methods, 1):
        order_id = f"order_{i:03d}"
        print(f"\n使用{method}支付订单{order_id}:")
        payment_result = process_payment_wrapper(manager)(order_id, method)
        print(payment_result)
    
    # 6. 测试清空购物车
    print("\n=== 测试清空购物车 ===")
    meituan_app.clear_cart()
    state, code = manager.get_device_state("meituan_app", "shopping_cart")
    if code == ErrorCode.SUCCESS:
        print("清空购物车后状态:")
        print(state)
    
    # 7. 测试设备特征信息
    print("\n=== 测试设备特征信息 ===")
    print(meituan_app.get_device_features_info())
    
    # 8. 测试环境信息收集
    print("\n=== 测试环境信息收集 ===")
    env_info = manager.collect_environment_info(format_type='natural_language', device_type=MeituanApp)
    print("\n美团App设备环境信息:")
    print(env_info)

def test_seat():
    # 创建设备管理器
    manager = DeviceManager()
    
    # 1. 创建并注册多个座椅设备
    print("=== 创建和注册座椅设备 ===")
    driver_seat = Seat(Seat.Position.DRIVER)
    passenger_seat = Seat(Seat.Position.PASSENGER)
    rear_left_seat = Seat(Seat.Position.REAR_LEFT)
    rear_right_seat = Seat(Seat.Position.REAR_RIGHT)
    
    seats = [driver_seat, passenger_seat, rear_left_seat, rear_right_seat]
    for seat in seats:
        result = manager.register_device(seat)
        print(f"注册{seat.device_id}: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 2. 测试座椅按摩功能
    print("\n=== 测试座椅按摩功能 ===")
    # 设置驾驶座按摩模式和强度
    print("\n设置驾驶座按摩：")
    result = manager.set_device_state("seat_driver", "massage_mode", "wave")
    print(f"设置按摩模式为wave: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    result = manager.set_device_state("seat_driver", "massage_intensity", Seat.MassageIntensity.MEDIUM)
    print(f"设置按摩强度为MEDIUM: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 设置后排座椅按摩
    print("\n设置后排左座按摩：")
    result = manager.set_device_state("seat_rear_left", "massage_mode", "pulse")
    print(f"设置按摩模式为pulse: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    result = manager.set_device_state("seat_rear_left", "massage_intensity", Seat.MassageIntensity.HIGH)
    print(f"设置按摩强度为HIGH: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 3. 测试座椅通风功能
    print("\n=== 测试座椅通风功能 ===")
    # 设置驾驶座通风
    result = manager.set_device_state("seat_driver", "ventilation_intensity", Seat.VentilationIntensity.HIGH)
    print(f"设置驾驶座通风强度为HIGH: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 设置副驾驶座通风
    result = manager.set_device_state("seat_passenger", "ventilation_intensity", Seat.VentilationIntensity.MEDIUM)
    print(f"设置副驾驶座通风强度为MEDIUM: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 4. 测试座椅占用状态
    print("\n=== 测试座椅占用状态 ===")
    # 设置多个座椅的占用状态
    occupancy_settings = {
        "seat_driver": True,
        "seat_passenger": True,
        "seat_rear_left": False,
        "seat_rear_right": True
    }
    
    for seat_id, is_occupied in occupancy_settings.items():
        result = manager.set_device_state(seat_id, "occupancy", is_occupied)
        print(f"设置{seat_id}占用状态为{is_occupied}: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 5. 获取所有座椅状态
    print("\n=== 获取所有座椅状态 ===")
    for seat_id in ["seat_driver", "seat_passenger", "seat_rear_left", "seat_rear_right"]:
        state, code = manager.get_device_state(seat_id)
        if code == ErrorCode.SUCCESS:
            print(f"\n{seat_id}当前状态:")
            print(f"- 占用状态: {state['occupancy']}")
            print(f"- 按摩模式: {state['massage_mode']}")
            print(f"- 按摩强度: {state['massage_intensity'].name}")
            print(f"- 通风强度: {state['ventilation_intensity'].name}")
    
    # 6. 测试批量操作
    print("\n=== 测试批量操作 ===")
    # 关闭所有座椅的按摩功能
    results = manager.batch_set_state(Seat, "massage_intensity", Seat.MassageIntensity.OFF)
    print("\n批量关闭按摩功能结果:")
    for seat_id, result in results.items():
        print(f"{seat_id}: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 关闭所有座椅的通风功能
    results = manager.batch_set_state(Seat, "ventilation_intensity", Seat.VentilationIntensity.OFF)
    print("\n批量关闭通风功能结果:")
    for seat_id, result in results.items():
        print(f"{seat_id}: {'成功' if result == ErrorCode.SUCCESS else '失败'}")
    
    # 7. 测试设备特征信息
    print("\n=== 测试设备特征信息 ===")
    print("\n座椅设备功能描述:")
    print(driver_seat.get_device_features_info())
    
    # 8. 测试环境信息收集
    print("\n=== 测试环境信息收集 ===")
    env_info = manager.collect_environment_info(format_type='natural_language', device_type=Seat)
    print("\n座椅设备环境信息:")
    print(env_info)

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

def process_payment_wrapper(device_manager: DeviceManager):
    def process_payment(order_id: str, payment_method: str = "微信支付") -> str:
        """
        处理订单支付（模拟支付过程）。
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

if __name__ == "__main__":
    print("=== 测试设备管理器 ===\n")
    print("=== 测试座椅设备 ===\n")
    test_seat()
    print("=== 测试美团APP ===\n")
    test_meituan_app()
