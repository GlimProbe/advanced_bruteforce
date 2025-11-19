# advanced_bruteforce
带验证码登录页面爆破

<h4>使用前提</h4>
1、安装Tesseract （UB-Mannheim 版）下载地址：<br>
https://github.com/UB-Mannheim/tesseract/releases<br>
安装最新版就行<br>
修改代码中对应位置为你的Tesseract目录<br>
<img width="558" height="101" alt="image" src="https://github.com/user-attachments/assets/7fefe338-0ffc-4e9e-8281-b0acf7d832cb" /><br>
若不会修改，安装完成Tesseract后将整个Tesseract-OCR目录复制到和advanced_bruteforce.py同一目录，如下：<br>
<img width="276" height="98" alt="image" src="https://github.com/user-attachments/assets/52073f2a-6148-43d6-8424-06b1c36ea29a" /><br>
<br>
2、安装依赖<br>
pip install requests pillow pytesseract opencv-python<br>
<br>
<h4>常见问题与调参建议</h4>
(1)识别经常是空串：<br>
    把 --ocr-timeout 提到 2.0～3.0，把 --ocr-budget 提到 2.0～3.0（脚本内也有自动加码与兜底）。<br>
(2)网关 502/过载：<br>
    --no-proxy；<br>
    --delay 0.25～0.5；<br>
    降低线程；<br>
    把 ocr-level 调到 fast。<br>
(3)固定长度验证码：<br>
    务必设对 --len-min/--len-max（如 4/4），投票能更准。<br>
<br>
<h4>用法</h4>
使用python advanced_bruteforce.py -h 查看参数。<br>
--parallel: 设置以后用户名字典和密码字典采用同行配对进行爆破，不会交叉配对。（就是用户名字典第一个对应密码字典第一个，不会和其他的配对），默认为交叉模式，不需要设置。<br>
--no-proxy: 不适用本地代理。<br>
--len-min/--len-max：设为相同，规定验证码长度。（如：--len-min 4 --len-max 4）<br>
--delay：每次尝试启动前的节拍延时。（默认0.15，一般不用改）<br>
--ocr-level：设置OCR强度档。（fast,balanced(默认),max）<br>
...<br>
其他就自己使用-h查看

