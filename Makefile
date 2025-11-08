# non-default targets:
# c: clean cmake files and fluid source, preserve libraries
# cm: clean cmake files
# ca: clean all (nuke build dir)
# r: run the program


.PHONY: default
default:
	mkdir -p build
	cd build && \
	cmake .. && \
	make
# for windows w/ mingw, replace the cmake line above with: 	cmake -G "MinGW Makefiles" .. && \

.PHONY: c
c: cm
	rm -rf ./build/CMakeFiles/fluid.dir/src/

.PHONY: cm
cm:
	rm -f ./build/CMakeCache.txt
	rm -f ./build/cmake_install.cmake
	rm -f ./build/cmake_install.local.cmake
	rm -f ./build/packaging.json
	rm -f ./build/compile_commands.json


.PHONY: ca
ca:
	rm -rf ./build/
	
.PHONY: r
r: default
	./build/fluid
.PHONY: update
update:
	git submodule update --remote
	mkdir -p build
	cd build && \
		cmake .. && \
		make

