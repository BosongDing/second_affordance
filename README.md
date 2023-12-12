This is the baseline of the dataset second affordance /

the file system should have the following structure:/
workspace
|---build  # 编译配置
|
|---docs   # 项目文档
|
|---dist	# 产出目录
|
|---src    # 开发目录
|     |---common    # 公用模块
|     |---plugins    # JS插件
|     |---components    # 项目公用组件
|     |---assets    # 资源文件
|     |     |- images   # 图片
|     |     |- icons    # iconfont/svg
|     |     |- stylus   # stylus
|     |---pages    # 页面
|     |     |- user   # 用户业务模块
|     |     |   |- components	# 当前页面组件
|     |     |   |- index.vue	# 页面(index.html、index.css、index.js)
|     |---store/dao    # 状态管理/数据逻辑层
|     |---apis    # ajax 抽离
|     |---routes    # 路由
|     |---config    # 页面内部变量配置
|
|---static    # 不需编译的外部资源
|
|---mock   # 数据模拟
|
|---test	# 测试脚本
|
|---config	# 项目编译配置
