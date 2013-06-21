#ifndef _PRINT_CONTEXT_H
#define _PRINT_CONTEXT_H

#include <vector>
#include <iostream>

class PrintContext;
static std::ostream & operator<<(std::ostream & out, PrintContext & ctx);

class PrintContext {
public:
	PrintContext() {
		count.push_back(0); //top level context
	}
	void push() {
		count.push_back(0);
		type.push_back('u'); //unknown type
		ident.push_back(-1); //unknown id
	}

	void mark(lElementType typ, int id) {
		assert(type.size() > 0);
		type.back() = elemTypeToChar(typ);
		ident.back() = id;
		count.back() = 0;
	}
	void pop() {
		assert(type.size() > 0);
		count.pop_back();
		type.pop_back();
		ident.pop_back();
		count.back() += 1;
	}
private:
	char elemTypeToChar(lElementType typ) {
		switch(static_cast<int>(typ)) {
			case L_VERTEX : return 'v';
			case L_EDGE   : return 'e';
			case L_FACE   : return 'f';
			case L_CELL   : return'c';
			default       : return 'u';
		}
	}
	std::vector< int > count;
	std::vector< char > type;
	std::vector< int > ident;
	friend std::ostream & operator<<(std::ostream & out, PrintContext & ctx);
};

inline static std::ostream & operator<<(std::ostream & out, PrintContext & ctx) {
	out << ctx.count[0];
	for(unsigned int i = 0; i < ctx.type.size(); i++) {
		out << ":" << ctx.type[i] << ctx.ident[i] << "-" << ctx.count[i+1];
	}
	ctx.count.back() += 1;
	return out << " ";
}


#endif