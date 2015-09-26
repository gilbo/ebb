#ifndef LISZTFILEREADER_H_
#define LISZTFILEREADER_H_
#include "Common.h"
#include "LisztFormat.h"
#include "../liszt_runtime.h"

namespace MeshIO {

struct BoundarySetEntry {
	IOElemType type;
	id_t start;
	id_t end;
	std::string name;
};

#define ABORT(x) do {                           \
        fprintf(stderr,x);                      \
        abort();                                \
    } while(0)

class LisztFileReader {
public:
	LisztFileReader() : file(NULL) {}

	void init(const std::string & filename) {
        FILE *input = NULL;
        init(filename, &input);	
	}

	void init(const std::string & filename, FILE **input) {
    if(input != NULL) {
      if(*input == NULL){
		    file = FOPEN(filename.c_str(),"r+");
        *input = file;
      }else{
		    file = *input;
      }
    }
		if(!file) {
			perror(NULL);
			ABORT("file read error");
		}
		if(fread(&head,sizeof(LisztHeader),1,file) != 1) {
			perror(NULL);
			ABORT("failed to read header");
		}
		if(head.magic_number != LISZT_MAGIC_NUMBER) {
			ABORT("unexpected magic number, is this a liszt mesh?");
		}
	}


/*
	void init(const std::string & filename) {
		file = FOPEN(filename.c_str(),"r");
		if(!file) {
			perror(NULL);
			ABORT("file read error");
		}
		if(fread(&head,sizeof(LisztHeader),1,file) != 1) {
			perror(NULL);
			ABORT("failed to read header");
		}
		if(head.magic_number != LISZT_MAGIC_NUMBER) {
			ABORT("unexpected magic number, is this a liszt mesh?");
		}
	}
*/	
	const LisztHeader & header() const {
		return head;
	}
	void readString(file_ptr pos, char * buf) {
	    seek(pos);
        if(!fgets(buf,2048,file)) {
            perror(NULL);
            ABORT("failed to read string");
        }
	}
	//client responsible for freeing BoundarySet * with this.free
	BoundarySetEntry * boundaries() {
		seek(head.boundary_set_table);
		BoundarySet * b = new BoundarySet[head.nBoundaries];
		if(fread(b,sizeof(BoundarySet),head.nBoundaries,file) != head.nBoundaries) {
			perror(NULL);
			ABORT("boundary read failed");
		}
		
		BoundarySetEntry * entries = new BoundarySetEntry[head.nBoundaries];
		
		for(unsigned int i = 0; i < head.nBoundaries; i++) {
			entries[i].type = b[i].type;
			entries[i].start = b[i].start;
			entries[i].end = b[i].end;
			
			char buf[2048]; //TODO(zach): buffer overrun for large symbols
			readString(b[i].name_string,buf);
			entries[i].name = buf;
		}
		delete [] b;
		return entries;
	}
	
	//get all fes, client responsible for freeing result with this.free
	FileFacetEdge * facetEdges() {
		return facetEdges(0,head.nFE);
	}
	
	//get range of facet edges [start,end), client responsible for freeing result with this.free
	FileFacetEdge * facetEdges(id_t start, id_t end) {
		assert(start <= head.nFE);
		assert(end <= head.nFE);
		assert(start <= end);
		lsize_t nfes = end - start;
		seek(head.facet_edge_table + start * sizeof(FileFacetEdge));
		
		FileFacetEdge * fes = new FileFacetEdge[nfes];
		
		if(fread(fes,sizeof(FileFacetEdge),nfes,file) != nfes) {
			perror(NULL);
			ABORT("error reading facet edges");
		}
		
		return fes;
	}
	size_t numFields() {
	    uint32_t nFields;
	    seek(head.field_table_index);
	    if(fread(&nFields,sizeof(uint32_t),1,file) != 1) {
	        perror(NULL);
            ABORT("error reading field");
	    }
	    return nFields;
	}
	void loadField(size_t offset, void ** data, char ** name, MeshIO::IOElemType * pelemtype, char * datatype, size_t * pelemlength) {
	    seek(head.field_table_index + sizeof(FieldTableIndex) + sizeof(FileField)*offset);
	    FileField field;
	    if(fread(&field,sizeof(FileField),1,file) != 1) {
            perror(NULL);
            ABORT("error reading field");
        }
        char buf[2048];
        readString(field.name,buf);
        *name = strdup(buf);
        size_t length = 1;
        if (field.range.flags & LISZT_VEC_FLAG)
            length *= field.range.data[0];
        if (field.range.flags == LISZT_MAT_FLAG)
            length *= field.range.data[1];
        size_t elemsize = lMeshTypeSize(field.range.type);
        size_t nelems;
        MeshIO::IOElemType elemtype = field.domain;
        switch(elemtype) {
            case MeshIO::VERTEX_T: nelems = head.nV; break;
            case MeshIO::EDGE_T: nelems = head.nE; break;
            case MeshIO::FACE_T: nelems = head.nF; break;
            case MeshIO::CELL_T: nelems = head.nC; break;
            default: ABORT("elemtype is wrong");
        }
        *data = malloc(elemsize*length*nelems);
        *pelemlength = length;
        *pelemtype = elemtype;
        *datatype = field.range.type;
        seek(field.data);
        if(fread(*data,elemsize*length,nelems,file) != nelems) {
            perror(NULL);
            ABORT("error reading field data");
        }
	}

	void free(FileFacetEdge * fe) {
		delete [] fe;
	}
	void free(BoundarySetEntry * bs) {
		delete [] bs;
	}
	void free(PositionTable * pt) {
		delete [] ((double*) pt);
	}
	void close() {
		fclose(file);
		file = NULL;
	}
private:
	void seek(file_ptr loc) {
//    printf("reader loc: %u\n", loc);
		if(fseeko(file,loc,SEEK_SET)) {
			perror(NULL);
			ABORT("reader::error seeking stream");
		}
	}
	LisztHeader head;
	FILE * file;
};

#undef ABORT
} // namespace MeshIO
#endif /* LISZTFILEREADER_H_ */
