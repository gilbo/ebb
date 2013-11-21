extern "C" {
#include "swiftshim.h"
}
#include <SWIFT.h>

swift_scene_t *swift_create_scene() {
	return reinterpret_cast<swift_scene_t *>(new SWIFT_Scene);
}

bool swift_add_object(swift_scene_t *scene, const SWIFT_Real *vertices, const int *faces,
        int num_vertices, int num_faces, int *id, bool fixed) {
    return reinterpret_cast<SWIFT_Scene *>(scene)->Add_Object(vertices, faces, num_vertices,
            num_faces, *id, fixed);
}

void swift_delete_object(swift_scene_t *scene, int id) {
    reinterpret_cast<SWIFT_Scene *>(scene)->Delete_Object(id);
}

void swift_set_object_transformation(swift_scene_t *scene, int id, const SWIFT_Real *R,
        const SWIFT_Real *T) {
    reinterpret_cast<SWIFT_Scene *>(scene)->Set_Object_Transformation(id, R, T);
}

bool swift_query_intersection(swift_scene_t *scene, bool early_exit, int *num_pairs, int **oids) {
    return reinterpret_cast<SWIFT_Scene *>(scene)->Query_Intersection(early_exit,
            *num_pairs, oids);
}

