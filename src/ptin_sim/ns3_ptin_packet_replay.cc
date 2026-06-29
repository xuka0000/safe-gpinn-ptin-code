#include <ns3/core-module.h>
#include <ns3/mobility-module.h>
#include <ns3/propagation-module.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

struct NodeRow {
  std::string cnNodeId;
  double xM = 0.0;
  double yM = 0.0;
};

struct PacketRow {
  std::string packetId;
  double timeS = 0.0;
  std::string srcCnNode;
  std::string dstCnNode;
  double payloadBytes = 512.0;
  double deadlineMs = 250.0;
  std::string scenarioId;
};

static std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool quoted = false;
  for (char ch : line) {
    if (ch == '"') {
      quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(ch);
    }
  }
  out.push_back(cur);
  return out;
}

static double ToDouble(const std::string& value, double fallback) {
  try {
    if (value.empty()) {
      return fallback;
    }
    return std::stod(value);
  } catch (...) {
    return fallback;
  }
}

static std::map<std::string, int> HeaderIndex(const std::vector<std::string>& header) {
  std::map<std::string, int> index;
  for (size_t i = 0; i < header.size(); ++i) {
    index[header[i]] = static_cast<int>(i);
  }
  return index;
}

static std::string Get(const std::vector<std::string>& fields, const std::map<std::string, int>& index, const std::string& name) {
  auto it = index.find(name);
  if (it == index.end() || it->second < 0 || static_cast<size_t>(it->second) >= fields.size()) {
    return "";
  }
  return fields[static_cast<size_t>(it->second)];
}

static std::map<std::string, NodeRow> ReadNodes(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::map<std::string, NodeRow> nodes;
  if (!std::getline(in, line)) {
    return nodes;
  }
  auto index = HeaderIndex(SplitCsvLine(line));
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto fields = SplitCsvLine(line);
    NodeRow row;
    row.cnNodeId = Get(fields, index, "cn_node_id");
    row.xM = ToDouble(Get(fields, index, "x_m"), 0.0);
    row.yM = ToDouble(Get(fields, index, "y_m"), 0.0);
    if (!row.cnNodeId.empty()) {
      nodes[row.cnNodeId] = row;
    }
  }
  return nodes;
}

static std::vector<PacketRow> ReadPackets(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::vector<PacketRow> packets;
  if (!std::getline(in, line)) {
    return packets;
  }
  auto index = HeaderIndex(SplitCsvLine(line));
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto fields = SplitCsvLine(line);
    PacketRow row;
    row.packetId = Get(fields, index, "packet_id");
    row.timeS = ToDouble(Get(fields, index, "time_s"), 0.0);
    row.srcCnNode = Get(fields, index, "src_cn_node");
    row.dstCnNode = Get(fields, index, "dst_cn_node");
    row.payloadBytes = ToDouble(Get(fields, index, "payload_bytes"), 512.0);
    row.deadlineMs = ToDouble(Get(fields, index, "deadline_ms"), 250.0);
    row.scenarioId = Get(fields, index, "scenario_id");
    if (!row.packetId.empty()) {
      packets.push_back(row);
    }
  }
  return packets;
}

static double DbmToMw(double dbm) {
  return std::pow(10.0, dbm / 10.0);
}

static double MwToDbm(double mw) {
  return 10.0 * std::log10(std::max(1.0e-30, mw));
}

static double MultipathFadingDb(double distanceM, double timeS) {
  return 4.0 * std::abs(std::sin(distanceM / 250.0 + timeS * 0.01));
}

static double CalcRxPowerDbm(Ptr<PropagationLossModel> loss, double txPowerDbm, const NodeRow& src, const NodeRow& dst) {
  Ptr<ConstantPositionMobilityModel> srcMob = CreateObject<ConstantPositionMobilityModel>();
  Ptr<ConstantPositionMobilityModel> dstMob = CreateObject<ConstantPositionMobilityModel>();
  srcMob->SetPosition(Vector(src.xM, src.yM, 0.0));
  dstMob->SetPosition(Vector(dst.xM, dst.yM, 0.0));
  return loss->CalcRxPower(txPowerDbm, srcMob, dstMob);
}

static double DistanceM(const NodeRow& src, const NodeRow& dst) {
  const double dx = src.xM - dst.xM;
  const double dy = src.yM - dst.yM;
  return std::sqrt(dx * dx + dy * dy);
}

static std::string ContentionKey(const PacketRow& packet) {
  std::ostringstream key;
  key << std::fixed << std::setprecision(3) << packet.timeS << "|" << packet.dstCnNode;
  return key.str();
}

int main(int argc, char** argv) {
  std::string nodesPath = "ns3_nodes.csv";
  std::string packetsPath = "ns3_packet_schedule.csv";
  std::string outputPath = "ns3_packet_results.csv";
  double txPowerDbm = 40.0;
  double rxThresholdDbm = -118.0;
  double dataRateMbps = 6.0;
  double noiseFloorDbm = -120.0;
  double sinrThresholdDb = 3.0;
  double interferenceScale = 0.0001;

  CommandLine cmd(__FILE__);
  cmd.AddValue("nodes", "CN node CSV", nodesPath);
  cmd.AddValue("packets", "Packet schedule CSV", packetsPath);
  cmd.AddValue("output", "Packet result CSV", outputPath);
  cmd.AddValue("txPowerDbm", "Transmit power in dBm", txPowerDbm);
  cmd.AddValue("rxThresholdDbm", "Receiver threshold in dBm", rxThresholdDbm);
  cmd.AddValue("dataRateMbps", "Data rate in Mbit/s", dataRateMbps);
  cmd.AddValue("noiseFloorDbm", "Noise floor in dBm", noiseFloorDbm);
  cmd.AddValue("sinrThresholdDb", "Minimum SINR in dB", sinrThresholdDb);
  cmd.AddValue("interferenceScale", "Cross-domain interference coupling factor", interferenceScale);
  cmd.Parse(argc, argv);

  auto nodes = ReadNodes(nodesPath);
  auto packets = ReadPackets(packetsPath);
  Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
  loss->SetPathLossExponent(2.7);
  loss->SetReference(1.0, 46.6777);

  std::map<std::string, int> packetsPerContentionDomain;
  std::map<std::string, int> packetRankPerContentionDomain;
  for (const auto& packet : packets) {
    packetsPerContentionDomain[ContentionKey(packet)] += 1;
  }

  std::ofstream out(outputPath);
  out << "packet_id,src_cn_node,dst_cn_node,time_s,distance_m,rx_power_dbm,path_loss_db,delay_ms,queue_delay_ms,contention_count,effective_data_rate_mbps,multipath_fading_db,sinr_db,delivered,drop_reason,scenario_id,truth_boundary\n";
  out << std::fixed << std::setprecision(6);

  for (const auto& packet : packets) {
    auto srcIt = nodes.find(packet.srcCnNode);
    auto dstIt = nodes.find(packet.dstCnNode);
    std::string drop = "delivered";
    bool delivered = true;
    double distanceM = 0.0;
    double rxPowerDbm = -999.0;
    double delayMs = 0.0;
    double queueDelayMs = 0.0;
    double effectiveDataRateMbps = dataRateMbps;
    double multipathFadingDb = 0.0;
    double sinrDb = -999.0;
    const std::string contentionKey = ContentionKey(packet);
    const int contentionCount = std::max(1, packetsPerContentionDomain[contentionKey]);
    const int queueRank = packetRankPerContentionDomain[contentionKey]++;

    if (srcIt == nodes.end() || dstIt == nodes.end()) {
      delivered = false;
      drop = "missing_cn_endpoint";
    } else {
      distanceM = DistanceM(srcIt->second, dstIt->second);
      multipathFadingDb = MultipathFadingDb(distanceM, packet.timeS);
      rxPowerDbm = CalcRxPowerDbm(loss, txPowerDbm, srcIt->second, dstIt->second) - multipathFadingDb;
      effectiveDataRateMbps = dataRateMbps / std::sqrt(static_cast<double>(contentionCount));
      const double propagationDelayMs = distanceM / 299792458.0 * 1000.0;
      const double serializationDelayMs = packet.payloadBytes * 8.0 / std::max(0.001, effectiveDataRateMbps * 1.0e6) * 1000.0;
      queueDelayMs = static_cast<double>(queueRank) * serializationDelayMs;
      delayMs = propagationDelayMs + serializationDelayMs + queueDelayMs;
      double interferenceMw = 0.0;
      for (const auto& other : packets) {
        if (other.packetId == packet.packetId || other.timeS != packet.timeS || ContentionKey(other) == contentionKey) {
          continue;
        }
        auto otherSrcIt = nodes.find(other.srcCnNode);
        if (otherSrcIt == nodes.end()) {
          continue;
        }
        const double otherDistanceM = DistanceM(otherSrcIt->second, dstIt->second);
        const double otherRxPowerDbm =
            CalcRxPowerDbm(loss, txPowerDbm, otherSrcIt->second, dstIt->second) -
            MultipathFadingDb(otherDistanceM, packet.timeS);
        interferenceMw += interferenceScale * DbmToMw(otherRxPowerDbm);
      }
      sinrDb = MwToDbm(DbmToMw(rxPowerDbm) / (DbmToMw(noiseFloorDbm) + interferenceMw));
      if (rxPowerDbm < rxThresholdDbm) {
        delivered = false;
        drop = "rx_power_below_threshold";
      } else if (sinrDb < sinrThresholdDb) {
        delivered = false;
        drop = "sinr_below_threshold";
      } else if (delayMs > packet.deadlineMs) {
        delivered = false;
        drop = "deadline_miss";
      }
    }

    Simulator::Schedule(Seconds(packet.timeS), []() {});
    out << packet.packetId << ","
        << packet.srcCnNode << ","
        << packet.dstCnNode << ","
        << packet.timeS << ","
        << distanceM << ","
        << rxPowerDbm << ","
        << (txPowerDbm - rxPowerDbm) << ","
        << delayMs << ","
        << queueDelayMs << ","
        << contentionCount << ","
        << effectiveDataRateMbps << ","
        << multipathFadingDb << ","
        << sinrDb << ","
        << (delivered ? 1 : 0) << ","
        << drop << ","
        << packet.scenarioId << ","
        << "ns3_log_distance_contention_queueing_fading_packet_feedback_not_full_wifi_mac"
        << "\n";
  }

  Simulator::Run();
  Simulator::Destroy();
  return 0;
}
