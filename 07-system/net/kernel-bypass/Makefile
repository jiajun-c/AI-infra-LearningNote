PKGCONF ?= pkg-config
TARGET := dpdk_udp_echo
SRC := dpdk_udp_echo.c

CFLAGS += -O3 -Wall -Wextra -Wpedantic

DPDK_CFLAGS := $(shell $(PKGCONF) --cflags libdpdk 2>/dev/null)
DPDK_LIBS := $(shell $(PKGCONF) --libs libdpdk 2>/dev/null)

.PHONY: all clean check-dpdk

all: check-dpdk $(TARGET)

check-dpdk:
	@$(PKGCONF) --exists libdpdk || \
		(echo "DPDK development package not found: pkg-config --exists libdpdk failed"; exit 1)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(DPDK_CFLAGS) $< -o $@ $(DPDK_LIBS)

clean:
	rm -f $(TARGET)
