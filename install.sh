#!/bin/bash
set -e

echo "=========================================="
echo "  课程伴学助教 - 一键安装脚本"
echo "=========================================="

# 更新系统
echo "[1/5] 更新系统..."
apt update -y

# 安装Python和pip
echo "[2/5] 安装Python环境..."
apt install -y python3 python3-pip python3-venv

# 创建应用目录
echo "[3/5] 创建应用目录..."
mkdir -p /opt/course-assistant
cd /opt/course-assistant

# 克隆代码
echo "[4/5] 下载应用代码..."
if [ -d ".git" ]; then
    git pull
else
    git clone https://github.com/zhaoyan19780204/course-assistant.git .
fi

# 创建虚拟环境并安装依赖
echo "[5/5] 安装Python依赖..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 创建systemd服务
echo "创建系统服务..."
cat > /etc/systemd/system/course-assistant.service << 'EOF'
[Unit]
Description=Course Assistant Streamlit App
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/course-assistant
Environment="PATH=/opt/course-assistant/venv/bin"
ExecStart=/opt/course-assistant/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
systemctl daemon-reload
systemctl enable course-assistant
systemctl start course-assistant

# 获取公网IP
PUBLIC_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "YOUR_SERVER_IP")

echo ""
echo "=========================================="
echo "  ✅ 安装完成！"
echo "=========================================="
echo ""
echo "访问地址: http://$PUBLIC_IP:8501"
echo ""
echo "常用命令:"
echo "  查看状态: systemctl status course-assistant"
echo "  查看日志: journalctl -u course-assistant -f"
echo "  重启服务: systemctl restart course-assistant"
echo ""
