# Nombre del ejecutable
TARGET = MLP

# Compilador y banderas
CC = gcc
CFLAGS =  
LFLAGS= -lm

# Archivos fuente y objetos
SRCS = main.c mlp.c dataset.c


all: $(TARGET)

# Enlazado
$(TARGET):
	$(CC) $(SRCS) $(LFLAGS) $(CFLAGS) -o $@ 


# Limpiar
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean install
