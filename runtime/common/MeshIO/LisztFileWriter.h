#ifndef _LISZT_FILE_WRITER_H_
#define _LISZT_FILE_WRITER_H_

#include "common/MeshIO/Common.h"
#include "common/MeshIO/LisztFormat.h"

namespace MeshIO {

class BoundaryTable;
class LisztFileWriter {
public:
	LisztFileWriter() : boundaries(NULL), file(NULL) {}
	LisztHeader header;
	BoundaryTable* boundaries;
/*
	void init(const std::string & filename) {
		file = FOPEN(filename.c_str(),"w");
		if(!file) {	
			fprintf(stderr,"error opening output file: %s\n",filename.c_str());
			exit(1);
		}
		
	}
*/
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
			fprintf(stderr,"error opening output file: %s\n",filename.c_str());
			exit(1);
		}
	}

	void setEnd() {
		if(fseeko(file,0,SEEK_END)) {
			perror(NULL);
			assert(!"setEnd::error seeking stream");
		}
	}
	void setBeginData() {
		if(fseeko(file,sizeof(LisztHeader),SEEK_SET)) {
			perror(NULL);
			assert(!"setBeginData::error seeking stream");
		}
	}
	void setBeginHeader() {
		if(fseeko(file,0,SEEK_SET)) {
			perror(NULL);
			assert(!"setBeginHeader::error seeking stream");
		}
	}
	void setPosition(file_ptr loc) {
		if(fseeko(file,loc,SEEK_SET)) {
			perror(NULL);
			assert(!"setPosition::error seeking stream");
		}
	}
	void write(size_t sz, const void * data) {
		if(fwrite(data,sz,1,file) != 1) {
			perror(NULL);
			assert(!"write failure");
		}
	}
	
	off_t currentPosition() {
		return ftello(file);
	}	
					 
	template<typename T>
	void writeObject(const T * obj) {
		write(sizeof(T),obj);
	}
	template<typename T>
	void writeValue(const T & obj) {
		write(sizeof(T),&obj);
	}
	
	void writeString(const std::string & str) {
		size_t len = strlen(str.c_str());
		if(fwrite(str.c_str(),len + 1,1,file) != 1) {
			perror(NULL);
			assert(!"write error");
		}
	}
	
	void writeHeader() {
	    header.magic_number = LISZT_MAGIC_NUMBER;
		setBeginHeader();
		writeObject(&header);
	}

	void close() {
		fclose(file);
	}
	
private:
	FILE * file;
};
} // namespace MeshIO
#endif /* LISZT_FILE_WRITER_H_ */
