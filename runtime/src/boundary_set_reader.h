#ifndef _BOUNDARY_SET_READER_H
#define _BOUNDARY_SET_READER_H
#include <map>
#include <set>
#include <string>
#include <vector>
#include "MeshIO/LisztFileReader.h"

class BoundarySetReader {
private:
	typedef std::map< std::string, MeshIO::BoundarySetEntry *> name_table_t;
	
	MeshIO::BoundarySetEntry * entries;
	size_t size;
	name_table_t name_table;
	
public:
	typedef std::vector< std::pair<MeshIO::id_t, MeshIO::id_t> > range_list_t;
	//takes ownership of boundaries
	void init(size_t nB, MeshIO::BoundarySetEntry * boundaries) {
		size = nB;
		entries = boundaries;
		//construct name -> table entry mapping
		for(size_t i = 0; i < size; i++) {
			if(entries[i].name.size() > 0)
				name_table[entries[i].name] = &entries[i];
		}
	}
	
	
public: //mesh adapter needs to get the ranges directly
	bool addSet(MeshIO::IOElemType t, MeshIO::BoundarySetEntry * b, std::vector<MeshIO::id_t> * points) {
		if( (b->type & ~MeshIO::AGG_FLAG) != t) {
			printf("boundary set aggregate changed type, file format error %d, expected %d",t,b->type & ~MeshIO::AGG_FLAG);
			return false;
		}
		if(b->type & MeshIO::AGG_FLAG) {
			assert(b->start < size);
			assert(b->end < size);
			return addSet(t,&entries[b->start],points) && 
			       addSet(t,&entries[b->end],points);
		} else {
			//we represent start and end ranges together in the same list
			//by shifting everyting left 1 and making start ranging even
			//and end ranges odd.
			//this makes end ranges slightly greater than start ranges
			//making it easy to march though and extract non-overlapping ranges
			points->push_back(b->start << 1);
			points->push_back( (b->end << 1) | 1);
			return true;
		}
	}
	
	MeshIO::BoundarySetEntry * entry(const std::string & name) {
		name_table_t::iterator it = name_table.find(name);
		if(it == name_table.end())
			return NULL;
		else
			return it->second;
	}

	bool load( const char * name,
	           int * id,
	           MeshIO::IOElemType * value,
	           range_list_t * ranges, 
	           size_t * size) {
		*size = 0;
		MeshIO::BoundarySetEntry * e = entry(name);
		if(!e) {
			printf("warning: bounary set not in file, it will be initialized as empty: %s\n",name);
			*id = name_table.size();
			return true;
		}
		*id = (e - entries);
		MeshIO::IOElemType typ = (MeshIO::IOElemType)(e->type & ~MeshIO::AGG_FLAG);
		std::vector<MeshIO::id_t> points;
		if(!addSet(typ,e,&points)) {
			exit(1);
			return false;
		}
		
		std::sort(points.begin(),points.end());
		
		id_t depth = 0;
		id_t start = 0;
		for(std::vector<MeshIO::id_t>::iterator it = points.begin(),end = points.end(); 
		    it != end; 
		    it++) {
		    bool is_end = *it & 1;
			id_t n = *it >> 1;
			if(is_end) {
				depth--;
				if(depth == 0) {
					ranges->push_back(std::make_pair(start,n));
					if(size)
						*size += (n - start);
				}
			} else {
				if(depth == 0)
					start = n;
				depth++;
			}
		}
		assert(depth == 0);
		return true;
	}
};
#endif