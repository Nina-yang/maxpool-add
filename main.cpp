#include "max_pool_add.hpp"
#include "fused_op.hpp"
#include "fused_graph.hpp"
#include "utils.hpp"
#include <chrono>




using  milliseconds = std::chrono:: milliseconds;
using hclock = std::chrono::high_resolution_clock;

template<typename T>
bool check_equal (const Tensor<T>& gt, const Tensor<T>& res) {
    // std::cout << gt.B << ", " << gt.C << ", " << gt.H << ", " << gt.W << std::endl;
    // std::cout << res.B << ", " << res.C << ", " << res.H << ", " << res.W << std::endl;
    if (res.B != gt.B) return false;
    // std::cout << "aa1" << std::endl;
    if (res.C != gt.C) return false;
    // std::cout << "aa2" << std::endl;
    if (res.H != gt.H) return false;
    // std::cout << "aa3" << std::endl;
    if (res.W != gt.W) return false;
    // std::cout << "aa4" << std::endl;

    size_t size = gt.size();
    // std::cout << "gt size: " << size << ", res size: " << res.size() << std::endl;
    for (size_t i = 0; i < size; ++i) {
        if (gt.p[i] != res.p[i]) 
            return false;
    }  
    return true;

}


template<typename T>
bool test_from_torch(const char* path_a, 
const char* path_b, const char* path_c, int case_index, 
std::string& data_type,  milliseconds& overall_time,
                     T* p_a=NULL, T* p_b=NULL, T* p_c=NULL, T* p_res=NULL,
                     bool verbose=false) {
    // std::cout << "000" << std::endl;
    Tensor<T> a(path_a, p_a); // construct tensor from file
    // std::cout << "1111" << std::endl;
    Tensor<T> b(path_b, p_b); // construct tensor from file
    // std::cout << "2222" << std::endl;
    Tensor<T> c(path_c, p_c); // construct tensor from file
    // std::cout << "3333" << std::endl;

    // Tensor<T> a(path_a); // construct tensor from file
    // Tensor<T> b(path_b); // construct tensor from file
    // Tensor<T> c(path_c); // construct tensor from file


    auto t_start = hclock::now(); 
    // Tensor<T> res  = max_pool_add(a, b);
    Tensor<T> res(16, 8, 32, 32, p_res);
    // std::cout << "res at beginning: " << res.B << ", " << res.C << ", " << res.H << ", " << res.W << std::endl;
    Stride srcb_str = b.stride();
    size_t flag_b = srcb_str.B == 1? 0:1;
    size_t flag_c = srcb_str.C == 1? 0:1;
    size_t flag_h = srcb_str.H == 1? 0:1;
    size_t flag_w = srcb_str.C == 1? 0:1;
    fused_graph(a, b, res, flag_b, flag_c, flag_h, flag_w);
    auto t_end = hclock::now();
    overall_time += std::chrono::duration_cast< milliseconds>(t_end - t_start ); 
    
    // std::cout << "uuu" << std::endl;
    // bool is_pass_ = check_equal(c,res);
    

    bool is_pass = (c == res);
    if(verbose){
        std::cout << "test case " << case_index << ",\t\t"
        << "a: " << a.B << " x " << a.C << " x "  << a.H << " x " << a.W << ",\t\t"
        << "b: " << b.B << " x " << b.C << " x "  << b.H << " x " << b.W << ",\t\t"
        << "data type: " << data_type<< ",\t\t"
        << (is_pass? " passed": " not passed") << std::endl;
    }
    return is_pass;
}

void test_from_torch(const char * folder_path, bool verbose=false) { 

    auto get_case_index_from_name = [](std::string str) -> int {
            int first = str.find("_");
            int second  = str.find("_",first+1);
            std::string digit_str = str.substr(first+1, second - first - 1);
            int case_index = std::stoi(digit_str);
            return case_index;
    };

    auto get_data_type_from_name = [](std::string str)->std::string {
            int first = str.find("_");
            int second  = str.find("_",first+1);
            int third = str.find("_", second + 1);
            return str.substr(second+1, third - second -1);
    };


    std::vector<std::string> paths = list_dir(folder_path);
    std::sort(paths.begin(),paths.end(),[get_case_index_from_name](std::string a, std::string b) -> int{ //sort cases by index
       return get_case_index_from_name(a) < get_case_index_from_name(b);
    });

    std::string path_prefix = folder_path + std::string("/case_");
    bool is_pass;
    size_t pass_cnt = 0;
    size_t case_nums = paths.size() / 3;
    milliseconds overall_time(0);
    
    
    size_t size_a = 16*8*63*63;
    size_t size_b = 16*8*32*32;
    size_t size_c = 16*8*32*32;
    void * p_a = malloc(size_a * sizeof(double));
    void * p_b = malloc(size_b * sizeof(double));
    void * p_c = malloc(size_c * sizeof(double));
    void * p_res = malloc(size_c * sizeof(double));
    // case_nums
    for (int i = 0; i < case_nums ; i += 1) {
        std::string data_type = get_data_type_from_name(paths[i*3]);
        std::string path_a = path_prefix +
        std::to_string(i) + "_" + data_type + std::string("_a.bin"); // path of tensor a

        std::string path_b = path_prefix +
        std::to_string(i) + "_" + data_type  + std::string("_b.bin"); // path of tensor b

        std::string path_res = path_prefix +
        std::to_string(i) + "_" + data_type + std::string("_c.bin"); // path of tensor c
        
        if (data_type == "int32") {
            is_pass = test_from_torch<int>(path_a.c_str(), path_b.c_str(), path_res.c_str(),i, data_type, overall_time,(int*)p_a, (int*)p_b, (int*)p_c, (int*)p_res, verbose);
        } else if (data_type == "float32"){
            is_pass = test_from_torch<float>(path_a.c_str(), path_b.c_str(), path_res.c_str(),i, data_type, overall_time,(float*)p_a, (float*)p_b, (float*)p_c, (float*)p_res, verbose);
        } else if (data_type == "double"){
            is_pass = test_from_torch<double>(path_a.c_str(), path_b.c_str(), path_res.c_str(),i, data_type, overall_time,(double*)p_a, (double*)p_b, (double*)p_c, (double*)p_res, verbose);
        } else {
            std::cerr << "Not supported data type" << std::endl;
            abort();
        }
  
        if (is_pass) pass_cnt++;
        if (is_pass){ std::cout << i << std::endl; }
     
    }
    
    if(p_a){
        free(p_a);
    }
    if(p_b){
        free(p_b);
    }
    if(p_c){
        free(p_c);
    }
    if(p_res){
        free(p_res);
    }

    std::cout << "all cases : " << case_nums << ", passed cases : " << pass_cnt << std::endl;
    std::cout << "overall running time: " <<overall_time.count()<<"  milliseconds\n";

}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Please specify a path to the directory of test cases" << std::endl;
        exit(1);
    } else{
        test_from_torch(argv[1], argv[2]);
    }

    return 0;
}