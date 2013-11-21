/* Minimal-functionality C bindings for the C++ SWIFT library. */
typedef double SWIFT_Real;
typedef struct swift_scene_dummy swift_scene_t;

swift_scene_t *swift_create_scene();
bool swift_add_object(swift_scene_t *scene, const SWIFT_Real *vertices, const int *faces,
        int num_vertices, int num_faces, int *id, bool fixed = false);
void swift_delete_object(swift_scene_t *scene, int id);
void swift_set_object_transformation(swift_scene_t *scene, int id, const SWIFT_Real *R,
        const SWIFT_Real *T);
bool swift_query_intersection(swift_scene_t *scene, bool early_exit, int *num_pairs, int **oids);
