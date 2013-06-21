#ifndef LISZTFILEREADER_H_
#define LISZTFILEREADER_H_
#include "common/MeshIO/Common.h"
#include "common/MeshIO/LisztFormat.h"
#include "common/runtime_util.h"

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
			seek(b[i].name_string);
			if(!fgets(buf,2048,file)) {
				perror(NULL);
				ABORT("failed to read string");
			}
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

  unsigned char *fieldData (lElementType  key_type,
                            lType         val_type, 
                            size_t        val_length, 
                            const char   *key, 
                            id_t          start,
                            id_t          end) 
  {
      seek(head.field_table_index);
      FieldTableIndex index;

      bool is_position = (strcmp(key, "position") == 0);

      if (fread(&index, sizeof(index), 1, file) != 1) {
          perror(NULL);
          ABORT("error reading field data");
      }

      uint32_t nfields = index.num_fields;

      // TODO(crystal) error checking here in case nfields is corrupt/really big?
      FileField fields[nfields];
      if (fread(fields, sizeof(FileField), nfields, file) != nfields) {
          perror(NULL);
          ABORT("error reading field data");
      }

      int i, length;
      char m_val_type;
      for (i = 0; i < nfields; i++) {
         	char buf[2048]; // TODO(Crystal) potential for bug here
          seek(fields[i].name);
		if(!fgets(buf,2048,file)) {
			perror(NULL);
			ABORT("failed to read string");
		}

          // Verify a field with a name match
      //    printf(" %s, %s\n", buf, key);
          if (strcmp(buf, key) == 0) {

              // Check for matching domain types
          //    printf(" %d, %d\n", key_type, (lElementType) fields[i].domain);
              if (key_type != (lElementType) fields[i].domain)
                  continue;

              // Check for matching range types
              m_val_type = fields[i].range.type;
         //     printf(" %d, %d\n", m_val_type, val_type);
              if (m_val_type != val_type)
                  // For position, we can cast doubles down to floats if necessary 
                  // (This is here for legacy compatability with float positions)
                  if (!is_position || m_val_type != LISZT_DOUBLE || val_type != L_FLOAT)
                      continue;

              // Check for matching range lengths
              length = 1;
              if (fields[i].range.flags & LISZT_VEC_FLAG)
                  length *= fields[i].range.data[0];

              if (fields[i].range.flags == LISZT_MAT_FLAG)
                  length *= fields[i].range.data[1];

              // Final test - if lengths match, break out of the loop and
              // start copying the field data over
            //  printf(" %d, %d\n", length, val_length); 
              if (length == val_length)
                  break;
          }
      }
      
      // Failed to find field
      if (i == nfields)
      {
          perror(NULL);
//          printf("%s\n", key);
          ABORT("error finding field");
      }

      size_t   m_typesize = lMeshTypeSize(m_val_type);
      lsize_t  nelems     = (end == 0) ? fields[i].nElems : end - start;

      unsigned char *data = (unsigned char *) malloc(m_typesize * length * nelems);
      seek(fields[i].data + start * m_typesize * length);

      if (fread(data, m_typesize * length, nelems, file) != nelems) {
          perror(NULL);
          ABORT("Error copying field data");
      }
      
      // For position field, downcast doubles to floats if necessary
      if (is_position && m_val_type == LISZT_DOUBLE && val_type == L_FLOAT) {
          size_t r_typesize = lUtilTypeSize(val_type);
          double *m_data = (double *) data;
          float  *r_data = (float  *) malloc(r_typesize * length * nelems);
          for (int i = 0; i < nelems * length; i++)
              *(r_data + i)=*(m_data++);
          std::free(data);
          data = (unsigned char *) r_data;
      }

      return data;
  }

  unsigned char *fieldData (lElementType key_type, lType val_type, size_t val_length, const char *key) 
  {
      return fieldData(key_type, val_type, val_length, key, 0, 0);
  }


  uint32_t findFieldTable(FileField **fields)
  {
      seek(head.field_table_index);
      FieldTableIndex index;

      if (fread(&index, sizeof(index), 1, file) != 1) {
          perror(NULL);
          ABORT("error reading field data");
      }

      uint32_t nfields = index.num_fields;

      // TODO(crystal) error checking here in case nfields is corrupt/really big?
      *fields = (FileField*)malloc(sizeof(FileField)*nfields);
      if (fread(*fields, sizeof(FileField), nfields, file) != nfields) {
          perror(NULL);
          ABORT("error reading field data");
      }

		  return nfields;
  }

  file_ptr findFieldPositionInTable(const char   *key)
  {
      int i = findFieldIndex(key);
      if(i == -1) return -1;

      seek(head.field_table_index);
      FieldTableIndex index;

      if (fread(&index, sizeof(index), 1, file) != 1) {
          perror(NULL);
          ABORT("error reading field data");
      }

      uint32_t nfields = index.num_fields;

//      printf("READER.findFieldPositionInTable\n");
//      printf("   nfields %d i %d\n", nfields, i);
      // TODO(crystal) error checking here in case nfields is corrupt/really big?
      FileField fields[nfields];
      if (fread(fields, sizeof(FileField), i, file) != i) {
          perror(NULL);
          ABORT("error reading field data");
      }

		  return ftello(file);
  }

  int findField (const char   *key, FileField *field ) 
  {
      file_ptr loc = findFieldPositionInTable(key);
      if(loc == -1) return -1;
      seek(loc);
      fread(field, sizeof(FileField), 1, file);
      return 1;
  }

  int findFieldIndex (const char   *key) 
  {
      seek(head.field_table_index);
      FieldTableIndex index;

      if (fread(&index, sizeof(index), 1, file) != 1) {
          perror(NULL);
          ABORT("error reading field data");
      }

      uint32_t nfields = index.num_fields;

      // TODO(crystal) error checking here in case nfields is corrupt/really big?
      FileField fields[nfields];
      if (fread(fields, sizeof(FileField), nfields, file) != nfields) {
          perror(NULL);
          ABORT("error reading field data");
      }

      int i, length;
      char m_val_type;
      for (i = 0; i < nfields; i++) {
         	char buf[2048]; // TODO(Crystal) potential for bug here
          seek(fields[i].name);
		      if(!fgets(buf,2048,file)) {
			      perror(NULL);
			      ABORT("failed to read string");
		      }

          // Verify a field with a name match
          
          if (strcmp(buf, key) == 0) break;

      }
      
      // Failed to find field
      if (i == nfields)  return -1;
      
      return i;
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
