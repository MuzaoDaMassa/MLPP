CXX = g++
CXXFLAGS = -std=c++17 -g -I/usr/local/include/opencv4
LDFLAGS = -L/usr/local/lib
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio

SRCDIR = src
TESTDIR = tests
BINDIR = bin

MAIN = $(BINDIR)/main_exe
TESTS = $(BINDIR)/tests_exe

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

TEST_SOURCES := $(wildcard $(TESTDIR)/*.cpp)
TEST_OBJECTS := $(TEST_SOURCES:$(TESTDIR)/%.cpp=$(BINDIR)/%.o)

.PHONY: all clean compile_src compile_tests

all: compile_src compile_tests

compile_src: $(MAIN)

compile_tests: $(TESTS)

$(MAIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TESTS): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR)/%.o: $(TESTDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(BINDIR)/*.o

build_and_clean: all 
	rm -f $(BINDIR)/*.o
