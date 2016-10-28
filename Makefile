all:
	cd src; catkin_init_workspace
	catkin_make

clean:
	@- $(RM) -rf build devel
	@- $(RM) src/CMakeLists.txt