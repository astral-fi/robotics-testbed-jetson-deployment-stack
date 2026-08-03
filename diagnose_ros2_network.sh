#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# diagnose_ros2_network.sh
#
# Run this INSIDE the Docker container on your PC to verify ROS 2 DDS
# communication with the Jetson's AprilTag publisher container.
#
# Usage:
#   docker compose exec gsplat_stack bash /ros2_ws/diagnose_ros2_network.sh
#   — or —
#   docker compose exec gsplat_stack bash
#   ./diagnose_ros2_network.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  ROS 2 Cross-Machine Diagnostics (PC ↔ Jetson)              ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ── 1. Environment check ────────────────────────────────────────────────────
echo -e "${YELLOW}[1/7] Environment Variables${NC}"
echo "  ROS_DOMAIN_ID:        ${ROS_DOMAIN_ID:-NOT SET (defaults to 0)}"
echo "  RMW_IMPLEMENTATION:   ${RMW_IMPLEMENTATION:-NOT SET}"
echo "  CYCLONEDDS_URI:       ${CYCLONEDDS_URI:-(not set)}"
echo "  ROS_LOCALHOST_ONLY:   ${ROS_LOCALHOST_ONLY:-NOT SET (good — should be unset or 0)}"
echo ""

if [ "${ROS_LOCALHOST_ONLY}" = "1" ]; then
    echo -e "  ${RED}✗ PROBLEM: ROS_LOCALHOST_ONLY=1 blocks all cross-machine traffic!${NC}"
    echo -e "  ${RED}  Fix: Remove this variable or set to 0 in BOTH containers.${NC}"
    echo ""
fi

# ── 2. Network interfaces ──────────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Network Interfaces${NC}"
ip -4 addr show | grep -E "inet " | awk '{print "  " $NF ": " $2}'
echo ""

# ── 3. Ping the Jetson ─────────────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Jetson Connectivity${NC}"
JETSON_IP="${JETSON_IP:-}"
if [ -z "$JETSON_IP" ]; then
    echo -e "  ${YELLOW}⚠ JETSON_IP not set. Skipping ping test.${NC}"
    echo "  Tip: Run with JETSON_IP=<ip> to enable ping test"
else
    if ping -c 2 -W 1 "$JETSON_IP" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Jetson ($JETSON_IP) is reachable${NC}"
    else
        echo -e "  ${RED}✗ Cannot ping Jetson at $JETSON_IP${NC}"
    fi
fi
echo ""

# ── 4. DDS multicast check ─────────────────────────────────────────────────
echo -e "${YELLOW}[4/7] DDS Multicast Route${NC}"
MCAST_ROUTE=$(ip route show | grep -i "239.255" || true)
if [ -n "$MCAST_ROUTE" ]; then
    echo -e "  ${GREEN}✓ Multicast route exists: $MCAST_ROUTE${NC}"
else
    # Check if default route can handle multicast
    DEFAULT_IF=$(ip route | grep default | awk '{print $5}' | head -1)
    echo "  No explicit multicast route (using default interface: ${DEFAULT_IF:-unknown})"
    echo "  This is usually fine with network_mode:host"
fi
echo ""

# ── 5. ROS 2 node discovery ────────────────────────────────────────────────
echo -e "${YELLOW}[5/7] ROS 2 Node Discovery (waiting 3 seconds...)${NC}"
NODES=$(timeout 3 ros2 node list 2>/dev/null || echo "TIMEOUT")
if [ "$NODES" = "TIMEOUT" ] || [ -z "$NODES" ]; then
    echo -e "  ${RED}✗ No nodes discovered. Possible causes:${NC}"
    echo "    - Jetson container not running"
    echo "    - Different ROS_DOMAIN_ID on Jetson vs PC"
    echo "    - Firewall blocking UDP ports 7400-7500"
    echo "    - ROS_LOCALHOST_ONLY=1 on either machine"
else
    echo -e "  ${GREEN}✓ Discovered nodes:${NC}"
    echo "$NODES" | sed 's/^/    /'
fi
echo ""

# ── 6. Topic discovery ─────────────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Topic Discovery${NC}"
TOPICS=$(timeout 3 ros2 topic list 2>/dev/null || echo "TIMEOUT")
if [ "$TOPICS" = "TIMEOUT" ] || [ -z "$TOPICS" ]; then
    echo -e "  ${RED}✗ No topics discovered${NC}"
else
    echo -e "  ${GREEN}✓ Discovered topics:${NC}"
    echo "$TOPICS" | sed 's/^/    /'
    echo ""

    # Check specifically for the AprilTag topics we need
    echo -e "  ${CYAN}Checking expected AprilTag topics:${NC}"
    for i in 0 1 2 3; do
        TOPIC="/cam${i}/tag_detections"
        if echo "$TOPICS" | grep -q "$TOPIC"; then
            echo -e "    ${GREEN}✓ $TOPIC${NC}"
        else
            echo -e "    ${RED}✗ $TOPIC — not found${NC}"
        fi
    done

    # Check output topics from our pipeline
    echo ""
    echo -e "  ${CYAN}Checking pipeline output topics:${NC}"
    for TOPIC in "/robot/fused_pose" "/gsplat/raw_image" "/gsplat/rendered_stream/ffmpeg"; do
        if echo "$TOPICS" | grep -q "$TOPIC"; then
            echo -e "    ${GREEN}✓ $TOPIC${NC}"
        else
            echo -e "    ${YELLOW}○ $TOPIC — not yet active${NC}"
        fi
    done
fi
echo ""

# ── 7. Live data check on AprilTag topics ───────────────────────────────────
echo -e "${YELLOW}[7/7] Live Data Check (2 second sample per topic)${NC}"
ANY_DATA=false
for i in 0 1 2 3; do
    TOPIC="/cam${i}/tag_detections"
    # Try to get one message with a 2-second timeout
    MSG=$(timeout 2 ros2 topic echo "$TOPIC" --once 2>/dev/null || echo "NO_DATA")
    if [ "$MSG" != "NO_DATA" ] && [ -n "$MSG" ]; then
        HZ=$(timeout 3 ros2 topic hz "$TOPIC" --window 5 2>/dev/null | head -1 || echo "")
        echo -e "  ${GREEN}✓ $TOPIC — receiving data${NC}"
        echo "    $HZ"
        ANY_DATA=true
    else
        echo -e "  ${RED}✗ $TOPIC — no data received in 2s${NC}"
    fi
done

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
if $ANY_DATA; then
    echo -e "${GREEN}  ✓ Data is flowing! The pipeline should be working.${NC}"
    echo ""
    echo "  Next steps:"
    echo "    1. Check fused pose:   ros2 topic echo /robot/fused_pose"
    echo "    2. Check rendered img: ros2 topic hz /gsplat/raw_image"
    echo "    3. Check H.264 stream: ros2 topic hz /gsplat/rendered_stream/ffmpeg"
else
    echo -e "${RED}  ✗ No AprilTag data flowing. Debug checklist:${NC}"
    echo ""
    echo "  1. Verify SAME ROS_DOMAIN_ID on both machines:"
    echo "     Jetson:  echo \$ROS_DOMAIN_ID"
    echo "     PC:      echo \$ROS_DOMAIN_ID"
    echo ""
    echo "  2. Verify ROS_LOCALHOST_ONLY is NOT set to 1:"
    echo "     unset ROS_LOCALHOST_ONLY"
    echo ""
    echo "  3. Verify SAME RMW on both machines:"
    echo "     ros2 doctor --report | grep middleware"
    echo ""
    echo "  4. Open firewall for DDS (CycloneDDS uses UDP 7400-7500):"
    echo "     sudo ufw allow 7400:7500/udp"
    echo ""
    echo "  5. For CycloneDDS, try explicit unicast peers instead of multicast:"
    echo "     export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery>"
    echo "       <Peers><Peer address=\"JETSON_IP\"/></Peers>"
    echo "       </Discovery></Domain></CycloneDDS>'"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
