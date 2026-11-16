/* probe_v3_label.c - does v3 silently mislabel a typo? */
#define main main_v3
#include "knn_v3.c"
#undef main
int main(void) {
    printf("v3 label_from_string(\"Hazardus\") [typo]  = %d  (2 means Hazardous)\n", label_from_string("Hazardus"));
    printf("v3 label_from_string(\"banana\")           = %d  (2 means Hazardous)\n", label_from_string("banana"));
    printf("v3 label_from_string(\"clean\") [lowercase]= %d  (0 expected)\n", label_from_string("clean"));
    return 0;
}
