#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rte_byteorder.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_udp.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8192
#define MBUF_CACHE_SIZE 256
#define BURST_SIZE 32

static volatile sig_atomic_t keep_running = 1;

static void handle_signal(int signo)
{
    (void)signo;
    keep_running = 0;
}

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [EAL options] -- -p <dpdk-port-id>\n", prog);
}

static int parse_app_args(int argc, char **argv, uint16_t *port_id)
{
    int opt;

    *port_id = 0;
    while ((opt = getopt(argc, argv, "p:h")) != -1) {
        switch (opt) {
        case 'p':
            *port_id = (uint16_t)strtoul(optarg, NULL, 10);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return -1;
        }
    }

    return 0;
}

static int port_init(uint16_t port_id, struct rte_mempool *mbuf_pool)
{
    struct rte_eth_conf port_conf;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_rxconf rxconf;
    struct rte_eth_txconf txconf;
    int ret;

    memset(&port_conf, 0, sizeof(port_conf));

    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        fprintf(stderr, "rte_eth_dev_info_get failed: %s\n", strerror(-ret));
        return ret;
    }

    ret = rte_eth_dev_configure(port_id, 1, 1, &port_conf);
    if (ret < 0) {
        fprintf(stderr, "rte_eth_dev_configure failed: %s\n", strerror(-ret));
        return ret;
    }

    rxconf = dev_info.default_rxconf;
    ret = rte_eth_rx_queue_setup(port_id, 0, RX_RING_SIZE,
                                 rte_eth_dev_socket_id(port_id),
                                 &rxconf, mbuf_pool);
    if (ret < 0) {
        fprintf(stderr, "rte_eth_rx_queue_setup failed: %s\n", strerror(-ret));
        return ret;
    }

    txconf = dev_info.default_txconf;
    ret = rte_eth_tx_queue_setup(port_id, 0, TX_RING_SIZE,
                                 rte_eth_dev_socket_id(port_id),
                                 &txconf);
    if (ret < 0) {
        fprintf(stderr, "rte_eth_tx_queue_setup failed: %s\n", strerror(-ret));
        return ret;
    }

    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        fprintf(stderr, "rte_eth_dev_start failed: %s\n", strerror(-ret));
        return ret;
    }

    ret = rte_eth_promiscuous_enable(port_id);
    if (ret != 0) {
        fprintf(stderr, "rte_eth_promiscuous_enable failed: %s\n", strerror(-ret));
        return ret;
    }

    return 0;
}

static int echo_udp_packet(struct rte_mbuf *mbuf)
{
    struct rte_ether_hdr *eth;
    struct rte_ipv4_hdr *ip;
    struct rte_udp_hdr *udp;
    struct rte_ether_addr tmp_mac;
    uint32_t tmp_ip;
    uint16_t tmp_port;
    uint16_t eth_type;
    uint8_t ip_header_len;

    if (rte_pktmbuf_data_len(mbuf) <
        sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr)) {
        return -1;
    }

    eth = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
    eth_type = rte_be_to_cpu_16(eth->ether_type);
    if (eth_type != RTE_ETHER_TYPE_IPV4) {
        return -1;
    }

    ip = (struct rte_ipv4_hdr *)(eth + 1);
    if (ip->version_ihl >> 4 != 4 || ip->next_proto_id != IPPROTO_UDP) {
        return -1;
    }

    ip_header_len = (uint8_t)((ip->version_ihl & 0x0f) * 4);
    if (ip_header_len < sizeof(struct rte_ipv4_hdr)) {
        return -1;
    }

    if (rte_pktmbuf_data_len(mbuf) <
        sizeof(struct rte_ether_hdr) + ip_header_len + sizeof(struct rte_udp_hdr)) {
        return -1;
    }

    udp = (struct rte_udp_hdr *)((char *)ip + ip_header_len);

    tmp_mac = eth->src_addr;
    eth->src_addr = eth->dst_addr;
    eth->dst_addr = tmp_mac;

    tmp_ip = ip->src_addr;
    ip->src_addr = ip->dst_addr;
    ip->dst_addr = tmp_ip;

    tmp_port = udp->src_port;
    udp->src_port = udp->dst_port;
    udp->dst_port = tmp_port;

    ip->hdr_checksum = 0;
    ip->hdr_checksum = rte_ipv4_cksum(ip);

    udp->dgram_cksum = 0;
    udp->dgram_cksum = rte_ipv4_udptcp_cksum(ip, udp);

    return 0;
}

int main(int argc, char **argv)
{
    struct rte_mempool *mbuf_pool;
    struct rte_mbuf *rx_pkts[BURST_SIZE];
    struct rte_mbuf *tx_pkts[BURST_SIZE];
    uint16_t port_id;
    uint16_t nb_ports;
    int ret;

    ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "failed to initialize EAL\n");
    }
    argc -= ret;
    argv += ret;

    if (parse_app_args(argc, argv, &port_id) != 0) {
        return EXIT_FAILURE;
    }

    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0 || port_id >= nb_ports) {
        rte_exit(EXIT_FAILURE, "invalid port %u, available ports: %u\n", port_id, nb_ports);
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    mbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", NUM_MBUFS, MBUF_CACHE_SIZE,
                                        0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                        rte_socket_id());
    if (mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "failed to create mbuf pool\n");
    }

    if (port_init(port_id, mbuf_pool) != 0) {
        rte_exit(EXIT_FAILURE, "failed to initialize port %u\n", port_id);
    }

    printf("DPDK UDP echo is running on port %u. Press Ctrl+C to stop.\n", port_id);

    while (keep_running) {
        uint16_t rx_count;
        uint16_t tx_count = 0;
        uint16_t sent;
        uint16_t i;

        rx_count = rte_eth_rx_burst(port_id, 0, rx_pkts, BURST_SIZE);
        if (rx_count == 0) {
            rte_pause();
            continue;
        }

        for (i = 0; i < rx_count; i++) {
            if (echo_udp_packet(rx_pkts[i]) == 0) {
                tx_pkts[tx_count++] = rx_pkts[i];
            } else {
                rte_pktmbuf_free(rx_pkts[i]);
            }
        }

        sent = 0;
        while (sent < tx_count) {
            uint16_t n = rte_eth_tx_burst(port_id, 0, &tx_pkts[sent], tx_count - sent);
            if (n == 0) {
                rte_pause();
                continue;
            }
            sent += n;
        }
    }

    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
    printf("stopped\n");

    return EXIT_SUCCESS;
}
