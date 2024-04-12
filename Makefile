CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

SRCDIR = src
TESTDIR = tests
BINDIR = bin

MAIN = $(BINDIR)/main_exe
TESTS = $(BINDIR)/tests_exe

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

TEST_SOURCES := $(wildcard $(TESTDIR)/*.cpp)
TEST_OBJECTS := $(TEST_SOURCES:$(TESTDIR)/%.cpp=$(BINDIR)/%.o)

.PHONY: all clean

all: $(MAIN) $(TESTS)

$(MAIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TESTS): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR)/%.o: $(TESTDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(BINDIR)/*.o

build_and_clean: all 
	rm -f $(BINDIR)/*.o
