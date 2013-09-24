#include "common/mesh.h"

//class used to wrap initialization of a CRS
class CRSInit {
public:
        CRSInit(const size_t & sz, CRS & c) : size(sz), crs(c) {
                set_sizes = new uint32_t[size];
                bzero(set_sizes,size *sizeof(uint32_t));
        }
        void found(uint32_t id) {
                assert(id < size);
                set_sizes[id]++;
        }
        void writeIndices() {
                initIndices();
                sizes_to_indices();
                initValues();
        }
        void add(uint32_t e, uint32_t v) {
                assert(e < size);
                assert(set_sizes[e] != 0);
                uint32_t idx = crs.row_idx[e+1] - set_sizes[e];
                crs.values[idx] = v;
                set_sizes[e]--;
        }
        void assert_finished_and_remove_duplicates() {
                size_check();
                for(unsigned int i = 0; i < size; i++) {
                        uint32_t * start = crs.values + crs.row_idx[i];
                        uint32_t * end = crs.values + crs.row_idx[i+1];
                        std::sort(start,end);
                        uint32_t * last = std::unique(start,end);
                        set_sizes[i] = last - start;
                }

                uint32_t * old_values = crs.values;
                //we now allocate space for the non-dup values
                uint32_t nelems = crs.row_idx[size];
                crs.values = (uint32_t*) malloc(nelems * sizeof(uint32_t));
                uint32_t * elem = crs.values;
                //copy the values over
                for(unsigned int i = 0; i < size; i++) {
                        uint32_t * begin = old_values + crs.row_idx[i];
                        for(unsigned int j = 0; j < set_sizes[i]; j++, elem++) {
                                *elem = begin[j];
                        }
                }
                //write the new indices into row_idx
                sizes_to_indices();
                //free the old values and clean up the initializer
                free(old_values);
                delete [] set_sizes;
                set_sizes = NULL;
        }
        void assert_finished() {
                size_check();
                delete [] set_sizes;
                set_sizes = NULL;
        }
        ~CRSInit() {
                assert(set_sizes == NULL);
        }
private:
        void initIndices() {
                crs.row_idx = (uint32_t *) malloc( (size + 1) * sizeof(uint32_t));
        }
        void initValues() {
                uint32_t nvalues = crs.row_idx[size];
                crs.values = (uint32_t *) malloc(nvalues * sizeof(uint32_t));
        }
        void sizes_to_indices() {
                crs.row_idx[0] = 0;
                // Index is based on running total of sizes
                for(unsigned int i = 1; i <= size; i++) {
                        crs.row_idx[i] = crs.row_idx[i-1] + set_sizes[i-1];
                }
        }
        void size_check() {
#ifndef NDEBUG
                for(unsigned int i = 0; i < size; i++) {
                        assert(set_sizes[i] == 0);
                }
#endif
        }
        uint32_t * set_sizes;
        const size_t & size;
        CRS & crs;
};

inline static bool same_array(uint32_t * start1, id_t * start2, size_t size) {
	const size_t MAX_SIZE = 12; //this needs to be set to the maximum size of any set we are detecting, currently 12 for HEXA's edge count
	assert(size <= MAX_SIZE);
	uint32_t c1[MAX_SIZE];
	id_t c2[MAX_SIZE];
	std::copy(start1,start1 + size,c1);
	std::copy(start2,start2 + size,c2);
	std::sort(c1,c1+size);
	std::sort(c2,c2+size);
	if (!std::equal(c1, c1 + size, c2)) {
		std::cout << "Something is weird here, arrays are: " << std::endl;
		for(size_t j = 0; j < size; j++) {
			std::cout << "orig: " << c1[j] << " modified: " << c2[j] << "unsorted: orig: " << start1[j] << " modified: " << start2[j] << std::endl;
		}
		return false;
	}
	/*
	std::cout << "New vs. Old: " << std::endl;
	for(size_t j = 0; j < size; j++) {
		std::cout << "orig: " << start1[j] << " modified: " << start2[j] << std::endl;
	}*/
	return true;
}
//check if the cell has same adjacencies as v, e, and f arrays but possibly permuted order.
//If they do, then replace them with new order
inline static void replace_if_same(uint32_t cell, Mesh * data, id_t * v, id_t * e, id_t * f, id_t * c) {
	id_t * new_orders[] = { v, e, f, c};
	CRS * adjs[] = { &data->ctov, &data->ctoe, &data->ctof, &data->ctoc };
	for(int i = 0; i < 4; i++) {
		CRS * crs = adjs[i];
		uint32_t start = crs->row_idx[cell];
		uint32_t end = crs->row_idx[cell+1];
		size_t size = end - start;
		if(!same_array(crs->values + start,new_orders[i],size))
			return;
	}
	//we have a permutation of the original values, we can safely replace them now
	for(int i = 0; i < 4; i++) {
		CRS * crs = adjs[i];
		uint32_t start = crs->row_idx[cell];
		uint32_t end = crs->row_idx[cell+1];
		size_t size = end - start;
		std::copy(new_orders[i], new_orders[i] + size, crs->values + start);
	}
}

size_t size_of_mesh_set(CRS * crs, uint32_t id) {
	return crs->row_idx[id + 1] - crs->row_idx[id];
}


void lMeshInitFromFacetEdgeBuilder(Mesh * mesh, MeshIO::FacetEdgeBuilder * b) {
	typedef MeshIO::FacetEdge::HalfFacet HalfFacet;
	typedef MeshIO::lsize_t lsize_t;
#define FOR_EACH(it,type) \
    for(lsize_t _i = 0; _i < b->nElems[type]; _i++) { \
    	MeshIO::FacetEdgeBuilder::map_t::iterator __iter = b->elem_maps[type].find(_i); \
    	if(__iter == b->elem_maps[type].end()) continue; \
    	HalfFacet * it = __iter->second;
    	
#define END_FOR_EACH }

    lsize_t nV = b->nV();
    lsize_t nE = b->nE();
    lsize_t nF = b->nF();
    lsize_t nC = b->nC();

    mesh->nvertices = nV;
    mesh->nedges = nE;
    mesh->nfaces = nF;
    mesh->ncells = nC;


    
    //edge to vertex init
    
    lsize_t nEF[] = { nE, nF };
    CRSConst * etovi[2] = { &mesh->etov, &mesh->ftoc };
	
	CRSInit vtoei[2] = { CRSInit(nV,mesh->vtoe) , CRSInit(nC,mesh->ctof) };
    CRSInit vtovi[2] = { CRSInit(nV,mesh->vtov) , CRSInit(nC,mesh->ctoc) };
	
    CRSInit etofi[2] = { CRSInit(nE,mesh->etof), CRSInit(nF,mesh->ftoe) };
    CRSInit etoci[2] = { CRSInit(nE,mesh->etoc), CRSInit(nF,mesh->ftov) };
    CRSInit ctoei[2] = { CRSInit(nC,mesh->ctoe), CRSInit(nV,mesh->vtof) };
    
    CRSInit ctovi[2] = { CRSInit(nC,mesh->ctov), CRSInit(nV,mesh->vtoc) };
    
    bool mesh_is_degenerate[] = { false , false };
    
    for(int t = 0; t < 2; t++) {
    	
    	etovi[t]->values = (uint32_t (*)[2]) malloc(nEF[t] * sizeof(uint32_t) * 2);
		
		// First accumulate counts of adjacencies so that we can figure out offsets
		FOR_EACH(ee,t + 2) //edge or face
			id_t e = ee->edgeOrFace(t);
			id_t head = ee->vertOrCell(t);
			id_t tail = ee->flip()->vertOrCell(t);
			
			vtoei[t].found(head);
			vtoei[t].found(tail);
			
			vtovi[t].found(head);
			vtovi[t].found(tail);
			
			HalfFacet * ff = ee;
			do {
				id_t inside = ff->flip()->vertOrCell(!t);
				etofi[t].found(e);
	
				etoci[t].found(e);
				ctoei[t].found(inside);
				
				ctovi[t].found(inside);
				ctovi[t].found(inside);
			
				ff = ff->ccwAroundEF[t];
			} while(ff != ee);
		END_FOR_EACH
		
		// Assign offset ranges and allocate memory
		vtoei[t].writeIndices();
		vtovi[t].writeIndices();
		
		etofi[t].writeIndices();
		etoci[t].writeIndices();
		ctoei[t].writeIndices();
		ctovi[t].writeIndices();
		
		// Now that offsets and memory are allocated, actually record adjacencies
		FOR_EACH(ee,t + 2)
			id_t e = ee->edgeOrFace(t);
			id_t head = ee->vertOrCell(t);
			id_t tail = ee->flip()->vertOrCell(t);
			if(head == tail)
				mesh_is_degenerate[t] = true;
			vtoei[t].add(head,e);
			vtoei[t].add(tail,e);
			
			vtovi[t].add(head,tail);
			vtovi[t].add(tail,head);
			
			etovi[t]->values[e][0] = head;
			etovi[t]->values[e][1] = tail;
			
			HalfFacet * ff = ee;
			do {
				id_t f = ff->edgeOrFace(!t);
				id_t outside = ff->vertOrCell(!t);
				id_t inside = ff->flip()->vertOrCell(!t);
	
				id_t direction;
				HalfFacet * mface = b->edge_face_m()[!t][f];
				if(mface->vertOrCell(!t) != outside) {
					direction = FLIP_DIRECTION_BIT;
				} else {
					direction = 0;
					assert(mface->flip()->vertOrCell(!t) == inside);
				}
				etofi[t].add(e,direction | f);
				
				etoci[t].add(e,inside);
				ctoei[t].add(inside,e);
				
				ctovi[t].add(inside,head);
				ctovi[t].add(inside,tail);
				
				ff = ff->ccwAroundEF[t];
			} while(ff != ee);
		END_FOR_EACH
		vtoei[t].assert_finished();
		vtovi[t].assert_finished();
		etofi[t].assert_finished();
		etoci[t].assert_finished();
		ctovi[t].assert_finished_and_remove_duplicates();
    }
    for(int t = 0; t < 2; t++) {
    	if(mesh_is_degenerate[t]) {
    		std::cout << "mesh has faces/edges that have the same cell/vert on both sides, removing duplicate edges/faces" << std::endl;
    		ctoei[!t].assert_finished_and_remove_duplicates();
    	} else {
    		ctoei[!t].assert_finished();
    	}
    }
    
    //now we detect the cell types:
    for(size_t cell_id = 1; cell_id < mesh->ncells; cell_id++) {
        size_t nv = size_of_mesh_set(&mesh->ctov, cell_id);
        size_t ne = size_of_mesh_set(&mesh->ctoe, cell_id);
        size_t nf = size_of_mesh_set(&mesh->ctof, cell_id);
        
        //TETRA
        if(nv == 4 && ne == 6 && nf == 4) {
           //cout << "Found potential tet: " << cell_id << std::endl;
           HalfFacet * fe = b->cellMap()[cell_id];
           id_t v[4]; std::fill_n(v,4,-1);
           id_t e[6]; std::fill_n(e,6,-1);
           id_t f[4]; std::fill_n(f,4,-1);
           id_t c[4]; std::fill_n(c,4,-1);
           
           v[0] = fe->vert();
           e[1] = fe->edge();
           f[1] = fe->face();
           c[1] = fe->flip()->cell();
           
           HalfFacet * fe2 = fe->ccwAroundEdge();
           f[2] = fe2->face();
           c[2] = fe2->cell();
           e[2] = fe2->ccwAroundFace()->edge();
           v[3] = fe2->ccwAroundFace()->vert();
           
           fe = fe->ccwAroundFace();
           v[1]= fe->vert();
           e[0] = fe->edge();
           fe2 = fe->ccwAroundEdge();
           f[0] = fe2->face();
           c[0] = fe2->cell();
           e[4] = fe2->ccwAroundFace()->edge();
           
           fe = fe->ccwAroundFace();
           v[2] = fe->vert();
           e[3] = fe->edge();
           fe2 = fe->ccwAroundEdge();
           f[3] = fe2->face();
           c[3] = fe2->cell();
           e[5] = fe2->ccwAroundFace()->edge();
           replace_if_same(cell_id,mesh,v,e,f,c);
        } else if(nv == 8 && ne == 12 && nf == 6) {
            //HEXA
            //cout << "Found potential hexa: " << cell_id << std::endl;
            HalfFacet * fe = b->cellMap()[cell_id];
            id_t v[8]; std::fill_n(v,8,-1);
            id_t e[12]; std::fill_n(e,12,-1);
            id_t f[6]; std::fill_n(f,6,-1);
            id_t c[6]; std::fill_n(c,6,-1);
            
            v[0] = fe->vert();
            e[1] = fe->edge();
            f[0] = fe->face();
            c[0] = fe->flip()->cell();
            HalfFacet * fe2 = fe->ccwAroundEdge();
            f[2] = fe2->face();
            c[2] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[4] = fe2->vert();
            e[2] = fe2->edge();
            e[9] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[1] = fe->vert();
            e[0] = fe->edge();
            fe2 = fe->ccwAroundEdge();
            f[1] = fe2->face();
            c[1] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[5] = fe2->vert();
            e[4] = fe2->edge();
            e[8] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[2] = fe->vert();
            e[3] = fe->edge();
            fe2 = fe->ccwAroundEdge();
            f[3] = fe2->face();
            c[3] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[6] = fe2->vert();
            e[5] = fe2->edge();
            e[10] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[3] = fe->vert();
            e[6] = fe->edge();
            fe2 = fe->ccwAroundEdge();
            f[4] = fe2->face();
            c[4] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[7] = fe2->vert();
            e[7] = fe2->edge();
            e[11] = fe2->ccwAroundFace()->edge();
            fe2 = fe2->ccwAroundFace()->flip()->ccwAroundEdge();
            f[5]  = fe2->face();
            c[5] = fe2->cell();
            replace_if_same(cell_id,mesh,v,e,f,c);
        } else if(nv == 6 && ne == 9 && nf == 5) {
            //WEDGE
            id_t v[6]; std::fill_n(v,6,-1);
            id_t e[9]; std::fill_n(e,9,-1);
            id_t f[5]; std::fill_n(f,5,-1);
            id_t c[5]; std::fill_n(c,5,-1);
            //cout << "Found potential wedge: " << cell_id << std::endl;
            //find a triangle face to start with
            
            uint32_t * faces = mesh->ctof.values + mesh->ctof.row_idx[cell_id];
            uint32_t face_id = faces[0];
            
            for(uint32_t i = 0; i < nf; i++) {
                if(size_of_mesh_set(&mesh->ftoe,faces[i]) == 3) {
                    face_id = faces[i];
                    break;
                }
            }
            
            if(size_of_mesh_set(&mesh->ftoe,face_id) != 3) {
                std::cout << "This should be wedge shaped but i can't find a triangle" << std::endl;
                continue;
            }
            HalfFacet * fe = b->faceMap()[face_id];
            
            if(fe->cell() != cell_id)
                fe = fe->flip();
            assert(fe->cell() == cell_id);
            f[0] = fe->face();
            c[0] = fe->flip()->cell();
            v[0] = fe->vert();
            e[0] = fe->edge();
            HalfFacet * fe2 = fe->ccwAroundEdge();
            f[2] = fe2->face();
            c[2] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[3] = fe2->vert();
            e[2] = fe2->edge();
            e[6] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[2] = fe->vert();
            e[1] = fe->edge();
            fe2 =  fe->ccwAroundEdge();
            f[1] = fe2->face();
            c[1] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[5] = fe2->vert();
            e[5] = fe2->edge();
            e[7] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[1] = fe->vert();
            e[3] = fe->edge();
            fe2 =  fe->ccwAroundEdge();
            f[3] = fe2->face();
            c[3] = fe2->cell();
            fe2 = fe2->ccwAroundFace();
            v[4] = fe2->vert();
            e[4] = fe2->edge();
            e[8] = fe2->ccwAroundFace()->edge();
            fe2 = fe2->ccwAroundFace()->flip()->ccwAroundEdge();
            f[4] = fe2->face();
            c[4] = fe2->cell();
            replace_if_same(cell_id,mesh,v,e,f,c);
        } else if(nv == 5 && ne == 8 && nf == 5) {
            //PYRAMID
            id_t v[5]; std::fill_n(v,5,-1);
            id_t e[8]; std::fill_n(e,8,-1);
            id_t f[5]; std::fill_n(f,5,-1);
            id_t c[5]; std::fill_n(c,5,-1);
            //cout << "Found potential pyramid: " << cell_id << std::endl;
            //find the quad face to start with
            
            uint32_t * faces = mesh->ctof.values + mesh->ctof.row_idx[cell_id];
            uint32_t face_id = faces[0];
            for(int i = 0; i < nf; i++) {
                if(size_of_mesh_set(&mesh->ftoe,faces[i]) == 4) {
                    face_id = faces[i];
                    break;
                }
            }
            if(size_of_mesh_set(&mesh->ftoe,face_id) != 4) {
                std::cout << "This should be pyramid shaped but i can't find the quad" << std::endl;
                continue;
            }
            HalfFacet * fe = b->faceMap()[face_id];
            
            if(fe->cell() != cell_id)
                fe = fe->flip();
            assert(fe->cell() == cell_id);
            
            f[0] = fe->face();
            c[0] = fe->flip()->cell();
            v[0] = fe->vert();
            e[1] = fe->edge();
            HalfFacet * fe2 = fe->ccwAroundEdge();
            f[1] = fe2->face();
            c[1] = fe2->cell();
            e[2] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[1] = fe->vert();
            e[0] = fe->edge();
            fe2 = fe->ccwAroundEdge();
            f[2] = fe2->face();
            c[2] = fe2->cell();
            e[4] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[2] = fe->vert();
            e[3] = fe->edge();
            fe2 = fe->ccwAroundEdge();
            f[3] = fe2->face();
            c[3] = fe2->cell();
            e[6] = fe2->ccwAroundFace()->edge();
            
            fe = fe->ccwAroundFace();
            v[3] = fe->vert();
            e[5] = fe->edge();
            fe2 = fe->ccwAroundEdge();
            f[4] = fe2->face();
            c[4] = fe2->cell();
            e[7] = fe2->ccwAroundFace()->edge();
            v[4] = fe2->ccwAroundFace()->vert();
            
            replace_if_same(cell_id,mesh,v,e,f,c);
        }
    } //forall cells
}
