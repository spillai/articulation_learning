REPO := git://github.com/PointCloudLibrary/pcl.git
CHECKOUT_DIR := pcl-1.7.0
REVISION := pcl-1.7.0rc2.1

default_target: all

# Default to a less-verbose build.  If you want all the gory compiler output,
# run "make VERBOSE=1"
$(VERBOSE).SILENT:

# Figure out where to build the software.
#   Use BUILD_PREFIX if it was passed in.
#   If not, search up to four parent directories for a 'build' directory.
#   Otherwise, use ./build.
ifeq "$(BUILD_PREFIX)" ""
BUILD_PREFIX:=$(shell for pfx in ./ .. ../.. ../../.. ../../../..; do d=`pwd`/$$pfx/build;\
               if [ -d $$d ]; then echo $$d; exit 0; fi; done; echo `pwd`/build)
endif
# create the build directory if needed, and normalize its path name
BUILD_PREFIX:=$(shell mkdir -p $(BUILD_PREFIX) && cd $(BUILD_PREFIX) && echo `pwd`)

# Default to a release build.  If you want to enable debugging flags, run
# "make BUILD_TYPE=Debug"
ifeq "$(BUILD_TYPE)" ""
BUILD_TYPE="Release"
endif

all: pod-build/Makefile
	$(MAKE) -C pod-build all install

pod-build/Makefile:
	$(MAKE) configure

.PHONY: configure
configure: $(CHECKOUT_DIR)/CMakeLists.txt
	@echo "\nBUILD_PREFIX: $(BUILD_PREFIX)\n\n"

	# create the temporary build directory if needed
	@mkdir -p pod-build

	cd $(CHECKOUT_DIR); \
	echo "Applying patches to cmake/CMakeLists file"; \
	patch -p0 -N -s -i ../pcl_targets_cmake.patch; \
	patch -p0 -N -s -i ../pcl_common_include_cmake.patch; \
	echo "Copying header cmake file"; \
	cp ../pkgconfig-headeronly.cmake.in cmake; \
	echo "Moving back to pod-directory"; \
	cd ../ ; \
	echo "Applied patches"; \

	# run CMake to generate and configure the build scripts
	@cd pod-build && cmake -DCMAKE_INSTALL_PREFIX=$(BUILD_PREFIX) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_surface=OFF \
		-DBUILD_geometry=ON ../$(CHECKOUT_DIR)
		   
$(CHECKOUT_DIR)/CMakeLists.txt:
	-if [ ! -d "$(CHECKOUT_DIR)" ];then \
		git clone $(REPO) $(CHECKOUT_DIR); \
		cd $(CHECKOUT_DIR); \
		git checkout $(REVISION); \
	fi

clean:
	-if [ -e pod-build/install_manifest.txt ]; then rm -f `cat pod-build/install_manifest.txt`; fi
	-if [ -d pod-build ]; then $(MAKE) -C pod-build clean; rm -rf pod-build; fi

# other (custom) targets are passed through to the cmake-generated Makefile 
%::
	$(MAKE) -C pod-build $@
