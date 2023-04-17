import gradio    
from gradio.core.schemas import MediaType    
from gradio.frontends.web import WebFrontend    
from gradio.frontends.web.inputs import WebCameraInput    
from gradio.frontends.web.outputs import WebVideoOutput    
from flask import Flask, request, render_template

app = Flask(__name__)

# 创建 WebFrontend 对象并添加摄像头输入和输出    
frontend = WebFrontend(url_prefix='/video', MediaType.VIDEO)    
frontend.add_input(WebCameraInput(prompt='Pick a camera'))    
frontend.add_output(WebVideoOutput())

# 注册路由以处理视频流请求    
@app.route('/', methods=['GET', 'POST'])    
def video_stream():    
    if request.method == 'POST':    
        # 获取摄像头输入的选择    
        camera_index = request.form.get('camera_index')    
        frontend.set_input(frontend.inputs[camera_index])    
        # 打开摄像头并获取视频流    
        stream = frontend.start_stream()    
        return stream    
    else:    
        # 默认 GET 请求，显示摄像头图标    
        return render_template('video.html')

if __name__ == '__main__':    
    app.run()    
