COMPILER=G++
C = cpp
OUTPUT_PATH = bin/
SOURCE_PATH = src/
EXE = $(OUTPUT_PATH)CV-Buoy
MKDIR_P = mkdir -p
OBJ = o
COPT = -O2
CCMD = g++
OBJFLAG = -o
EXEFLAG = -o
INCLUDES =
LIBS = `pkg-config --libs opencv`
LIBPATH =
CFLAGS = `pkg-config --cflags opencv`
CPPFLAGS = $(CFLAGS) $(COPT) -g $(INCLUDES)
LDFLAGS = $(LIBPATH) -g $(LIBS)
DEP = dep

OBJS := $(patsubst %.$(C),%.$(OBJ),$(wildcard $(SOURCE_PATH)*.$(C)))

%.$(OBJ):%.$(C)
	@echo Compiling $(basename $<)...
	$(CCMD) -c $(CPPFLAGS) $(CXXFLAGS) $< $(OBJFLAG)$@

all: $(OBJS)
	${MKDIR_P} ${OUTPUT_PATH}
	@echo Linking...
	$(CCMD) $(LDFLAGS) $^ $(LIBS) $(EXEFLAG) $(EXE)

clean:
	rm -rf $(SOURCE_PATH)*.$(OBJ) $(EXE)

rebuild: clean all
