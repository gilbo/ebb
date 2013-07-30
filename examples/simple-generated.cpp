struct lFields {
    lField Main___field;
    lField Main___position;
};
struct lkFields {
    lkField Main___field;
    lkField Main___position;
};
struct lScalars {
    lScalar com;
};
struct lkScalars {
    lkScalar com;
};
struct lSets {
};
L_UNNESTED void Main___main_unnested(lContext * __ctx );
L_UNNESTED void _liszt_init_globals_unnested(lContext * __ctx );
L_UNNESTED void _liszt_start_unnested(lContext * __ctx );
L_STENCIL void _liszt_init_globals_stencil(lsFunctionTable * __vtbl, lsContext * __ctx);
L_KERNEL void kernel_0(lkContext __ctx_struct);
L_STENCIL lStencilData kernel_0_stencil_data();
L_UNNESTED void Main___main_unnested(lContext * __ctx )
{
    float temp2 = 0.0;
    float temp3 = 0.0;
    float temp4 = 0.0;
    vec< 3 , float > com = {temp2,temp3,temp4};
    lScalarWrite(__ctx,&lGetScalars(__ctx)->com,L_ASSIGN,L_FLOAT,3,0,3,&com);
    lSet temp7;
    lVerticesOfMesh(__ctx,&temp7);
    lScalarEnterPhase(&lGetScalars(__ctx)->com,L_FLOAT,3,L_REDUCE_PLUS);
    lFieldEnterPhase(&lGetFields(__ctx)->Main___position,L_FLOAT,3,L_READ_ONLY);
    lKernelRun(__ctx, &temp7, L_VERTEX, 0, kernel_0, kernel_0_stencil_data());
    lSet temp9;
    lVerticesOfMesh(__ctx,&temp9);
    int temp10 = lSetSize(__ctx,&temp9);
    float temp11 = toFloat(temp10);
    lScalarEnterPhase(&lGetScalars(__ctx)->com,L_FLOAT,3,L_READ_ONLY);
    vec< 3 , float > tmp_1 = {temp11,temp11,temp11};
    lScalarWrite(__ctx,&lGetScalars(__ctx)->com,L_DIVIDE,L_FLOAT,3,0,3,&tmp_1);
    const char * temp12 = "\103\145\156\164\145\162\40\157\146\40\155\141\163\163\40\157\146\40\155\145\163\150\72\40";
    vec< 3 , float > com_2;
    lScalarRead(__ctx,&lGetScalars(__ctx)->com,L_FLOAT,3,0,3,&com_2);
    lPrintBegin(__ctx);
    lPrintValue(__ctx,L_STRING,0,0,temp12);
    lPrintValue(__ctx,L_FLOAT,3,0,&com_2);
    lPrintEnd(__ctx);
}
L_UNNESTED void _liszt_init_globals_unnested(lContext * __ctx )
{
    lScalarInit(__ctx,&lGetScalars(__ctx)->com,L_FLOAT,3);
    lFieldInit(__ctx,&lGetFields(__ctx)->Main___position,0,L_VERTEX,L_FLOAT,3);
    lFieldLoadData(__ctx,&lGetFields(__ctx)->Main___position,L_VERTEX,L_FLOAT,3,"\160\157\163\151\164\151\157\156");
    float Main___temp1 = 0.0;
    lFieldInit(__ctx,&lGetFields(__ctx)->Main___field,1,L_FACE,L_FLOAT,1);
    lFieldBroadcast(__ctx,&lGetFields(__ctx)->Main___field,L_FACE,L_FLOAT,1,&Main___temp1);
    _liszt_start_unnested(__ctx );
}
L_UNNESTED void _liszt_start_unnested(lContext * __ctx )
{
    Main___main_unnested(__ctx );
}
L_KERNEL void kernel_0(lkContext __ctx_struct)
{
    lkContext * __ctx = &__ctx_struct;
    lkElement v;
    if(lkGetActiveElement(__ctx,&v))
    {
        vec< 3 , float > temp5;
        lkFieldRead(&lkGetFields(__ctx)->Main___position,v,L_FLOAT,3,0,3,&temp5);
        lkScalarWrite(__ctx,&lkGetScalars(__ctx)->com,L_PLUS,L_FLOAT,3,0,3,&temp5);
    }
}
int main(int argc, char ** argv) {
    lProgramArguments arguments;
    lUtilParseProgramArguments(argc,argv,&arguments);
    lExec(_liszt_init_globals_unnested,_liszt_init_globals_stencil,&arguments,2,0,1);
}
L_STENCIL void kernel_0_stencil(lsFunctionTable * __vtbl, lsContext * __ctx)
{
    lsElement e0;

    __vtbl->lsGetActiveElement(__ctx,&e0);
    __vtbl->lsFieldAccess(__ctx,0,L_VERTEX,&e0,L_READ_ONLY);
}
L_STENCIL lStencilData kernel_0_stencil_data()
{
    lStencilData data = { kernel_0_stencil, true };
    return data;
}
L_STENCIL void _liszt_init_globals_stencil(lsFunctionTable * __vtbl, lsContext * __ctx)
{
    lsSet e1;
    __vtbl->lsVerticesOfMesh(__ctx,&e1);
    lsIterator it3;
    lsElement e2;
    __vtbl->lsSetGetIterator(__ctx, &e1, &it3);
    while(__vtbl->lsIteratorNext(__ctx,&it3, &e2)) {
        __vtbl->lsFieldAccess(__ctx,0,L_VERTEX,&e2,L_READ_ONLY);
    }
}
