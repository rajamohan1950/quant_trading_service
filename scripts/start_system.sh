#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Tick Generation & Latency Monitoring System${NC}"

# Kill any existing processes
echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
pkill -f "python.*websocket_server" 2>/dev/null
pkill -f "python.*kafka_consumer" 2>/dev/null  
pkill -f "python.*tick_generator" 2>/dev/null
pkill -f streamlit 2>/dev/null
sleep 2

# Check if Kafka is running
echo -e "${YELLOW}📊 Checking Kafka status...${NC}"
if ! docker-compose -f docker-compose.latency.yml ps | grep -q "Up"; then
    echo -e "${YELLOW}⚙️ Starting Kafka infrastructure...${NC}"
    docker-compose -f docker-compose.latency.yml up -d kafka zookeeper
    sleep 10
else
    echo -e "${GREEN}✅ Kafka already running${NC}"
fi

# Wait for Kafka to be ready
echo -e "${YELLOW}⏳ Waiting for Kafka to be ready...${NC}"
while ! docker exec kafka kafka-topics --bootstrap-server kafka:9092 --list >/dev/null 2>&1; do
    echo "Waiting for Kafka..."
    sleep 2
done
echo -e "${GREEN}✅ Kafka is ready${NC}"

# Start WebSocket server
echo -e "${YELLOW}🌐 Starting WebSocket server...${NC}"
python3 python_components/websocket_server.py &
WS_PID=$!
sleep 3

# Check if WebSocket server started
if ! ps -p $WS_PID > /dev/null; then
    echo -e "${RED}❌ WebSocket server failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}✅ WebSocket server started (PID: $WS_PID)${NC}"

# Start Kafka consumer
echo -e "${YELLOW}📥 Starting Kafka consumer...${NC}"
python3 python_components/kafka_consumer.py &
CONSUMER_PID=$!
sleep 3

# Check if Kafka consumer started
if ! ps -p $CONSUMER_PID > /dev/null; then
    echo -e "${RED}❌ Kafka consumer failed to start${NC}"
    kill $WS_PID 2>/dev/null
    exit 1
fi
echo -e "${GREEN}✅ Kafka consumer started (PID: $CONSUMER_PID)${NC}"

# Clean up database locks
echo -e "${YELLOW}🗄️ Cleaning database locks...${NC}"
rm -f stock_data.duckdb.lock

# Start Streamlit app
echo -e "${YELLOW}📱 Starting Streamlit application...${NC}"
python3 -m streamlit run app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!
sleep 5

# Check if Streamlit started
if ! ps -p $STREAMLIT_PID > /dev/null; then
    echo -e "${RED}❌ Streamlit failed to start${NC}"
    kill $WS_PID $CONSUMER_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}🎉 System started successfully!${NC}"
echo -e "${GREEN}📱 Streamlit app: http://localhost:8501${NC}"
echo -e "${GREEN}🌐 WebSocket server: ws://localhost:8080${NC}"
echo -e "${GREEN}📊 Kafka: localhost:9092${NC}"
echo ""
echo -e "${YELLOW}Process IDs:${NC}"
echo -e "  WebSocket server: $WS_PID"
echo -e "  Kafka consumer: $CONSUMER_PID" 
echo -e "  Streamlit app: $STREAMLIT_PID"
echo ""
echo -e "${YELLOW}To stop all processes:${NC}"
echo -e "  kill $WS_PID $CONSUMER_PID $STREAMLIT_PID"
echo ""
echo -e "${GREEN}✅ Ready for 10 million tick generation!${NC}"

# Keep script running and show status
while true; do
    sleep 30
    if ! ps -p $WS_PID > /dev/null; then
        echo -e "${RED}❌ WebSocket server died${NC}"
        break
    fi
    if ! ps -p $CONSUMER_PID > /dev/null; then
        echo -e "${RED}❌ Kafka consumer died${NC}"
        break
    fi
    if ! ps -p $STREAMLIT_PID > /dev/null; then
        echo -e "${RED}❌ Streamlit app died${NC}"
        break
    fi
    echo -e "${GREEN}💚 All systems running...${NC}"
done 