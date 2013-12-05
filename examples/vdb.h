#ifndef __VDB_H
#define __VDB_H

#define VDB_CALL

VDB_CALL int vdb_point(float x, float y, float z);
VDB_CALL int vdb_line(float x0, float y0, float z0, 
                      float x1, float y1, float z1);
VDB_CALL int vdb_normal(float x, float y, float z, 
                        float dx, float dy, float dz);
VDB_CALL int vdb_triangle(float x0, float y0, float z0, 
                          float x1, float y1, float z1,
                          float x2, float y2, float z2);

VDB_CALL int vdb_color(float r, float g, float b);

//By default, vdb will refresh the contents of the view at the end of every API call,
//you can surround multiple calls to vdb functions with vdb_begin/vdb_end
//to prevent the viewer from refreshing the contents until all the draw calls have been made

VDB_CALL int vdb_begin();
VDB_CALL int vdb_end();

VDB_CALL int vdb_flush();

//create a new blank frame. Currently just clears the screen, but eventually the viewer may keep
//around the contents of previous frames for inspection.
VDB_CALL int vdb_frame();


//versions that take direct pointers to floating point data
//this works well if you have a Point or Line struct
VDB_CALL int vdb_point_v(void * p);
VDB_CALL int vdb_line_v(void * p);
VDB_CALL int vdb_normal_v(void * p);
VDB_CALL int vdb_triangle_v(void * p);                 
VDB_CALL int vdb_color_v(void * c);



VDB_CALL int vdb_sample(float p);
VDB_CALL int vdb_label(const char * lbl); 
VDB_CALL int vdb_label_i(int i); 
#endif
