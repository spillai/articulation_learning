REPO := http://lcm.googlecode.com/svn/trunk
CHECKOUT_DIR := lcm

all: $(CHECKOUT_DIR)/Makefile
	$(MAKE) -C $(CHECKOUT_DIR) install

# Default to a less-verbose build.  If you want all the gory compiler output,
# run "make VERBOSE=1"
$(VERBOSE).SILENT:

# Figure out where to build the software.
#   Use BUILD_PREFIX if it was passed in.
#   If not, search up to four parent directories for a 'build' directory.
#   Otherwise, use ./build.
ifeq "$(BUILD_PREFIX)" ""
BUILD_PREFIX:=$(shell for pfx in .. ../.. ../../.. ../../../..; do d=`pwd`/$$pfx/build;\
               if [ -d $$d ]; then echo $$d; exit 0; fi; done; echo `pwd`/build)
endif
# create the build directory if needed, and normalize its path name
BUILD_PREFIX:=$(shell mkdir -p $(BUILD_PREFIX) && cd $(BUILD_PREFIX) && echo `pwd`)

# Default to a release build.  If you want to enable debugging flags, run
# "make BUILD_TYPE=Debug"
OPT_FLAGS := #-g -O2
ifeq "$(BUILD_TYPE)" "Debug"
OPT_FLAGS = -g
endif

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	MY_LDFLAGS=-L$(BUILD_PREFIX)/lib
else
	MY_LDFLAGS=-L$(BUILD_PREFIX)/lib -Wl,-R$(BUILD_PREFIX)/lib $(LDFLAGS)
endif

$(CHECKOUT_DIR)/Makefile:
	$(MAKE) configure

.PHONY: configure
configure: $(CHECKOUT_DIR)/configure
	# run configure
	@cd $(CHECKOUT_DIR)  && \
		./configure --prefix=$(BUILD_PREFIX) \
		PKG_CONFIG_PATH=$(PKG_CONFIG_PATH):$(BUILD_PREFIX)/lib/pkgconfig \
		CFLAGS="-I$(BUILD_PREFIX)/include $(OPT_FLAGS) $(CFLAGS)" \
		CXXFLAGS="-I$(BUILD_PREFIX)/include $(OPT_FLAGS) $(CXXFLAGS)" \
		LDFLAGS="$(MY_LDFLAGS)"  --disable-silent-rules

$(CHECKOUT_DIR)/configure:
	@svn checkout $(REPO) $(CHECKOUT_DIR)
	@echo "\nRunning bootstrap.sh \n\n"
	@cd $(CHECKOUT_DIR)  && ./bootstrap.sh
	@echo "\nBUILD_PREFIX: $(BUILD_PREFIX)\n\n"


clean:
	-if [ -e $(CHECKOUT_DIR)/Makefile ]; then $(MAKE) -C $(CHECKOUT_DIR) clean uninstall; fi
	-if [ -e $(CHECKOUT_DIR)/Makefile ]; then $(MAKE) -C $(CHECKOUT_DIR) distclean; fi

# other (custom) targets are passed through to the cmake-generated Makefile 
%::
	$(MAKE) -C pod-build $@
