#include "test.h"

bool test_activations() {

    const int SIZE = 64;
    const float EPS = 1e-3;

    std::vector<std::string> activation_names = {
        "elu",
        "hardtanh",
        "leakyrelu",
        "relu",
        "gelu",
        "gelu_approx_tanh",
        "sigmoid",
        "silu",
        "tanh",
        "softsign",
        "sin",
        "cos",
        "identity"
    };

    std::vector<bool> pass_activations;

    for (int i = 0; i < activation_names.size(); i++) {
        std::string activation_name = activation_names[i];

        float x_in_float[SIZE];
        float x_out_float[SIZE];

        load_data_1d<SIZE>(("./tb_data/test_activations_x_in_" + activation_name + ".bin").c_str(), x_in_float);
        load_data_1d<SIZE>(("./tb_data/test_activations_x_out_" + activation_name + ".bin").c_str(), x_out_float);

        F_TYPE x_in_fixed[SIZE];
        cast_1d<SIZE>(x_in_float, x_in_fixed);

        F_TYPE x_out_kernel_fixed[SIZE];

        for(int i = 0; i < SIZE; i++) {
            if (activation_name == "elu") x_out_kernel_fixed[i] = activation_elu(x_in_fixed[i]);
            if (activation_name == "hardtanh") x_out_kernel_fixed[i] = activation_hardtanh(x_in_fixed[i]);
            if (activation_name == "leakyrelu") x_out_kernel_fixed[i] = activation_leakyrelu(x_in_fixed[i]);
            if (activation_name == "relu") x_out_kernel_fixed[i] = activation_relu(x_in_fixed[i]);
            if (activation_name == "gelu") x_out_kernel_fixed[i] = activation_gelu(x_in_fixed[i]);
            if (activation_name == "gelu_approx_tanh") x_out_kernel_fixed[i] = activation_gelu_approx_tanh(x_in_fixed[i]);
            if (activation_name == "sigmoid") x_out_kernel_fixed[i] = activation_sigmoid(x_in_fixed[i]);
            if (activation_name == "silu") x_out_kernel_fixed[i] = activation_silu(x_in_fixed[i]);
            if (activation_name == "tanh") x_out_kernel_fixed[i] = activation_tanh(x_in_fixed[i]);
            if (activation_name == "softsign") x_out_kernel_fixed[i] = activation_softsign(x_in_fixed[i]);
            if (activation_name == "sin") x_out_kernel_fixed[i] = activation_sin(x_in_fixed[i]);
            if (activation_name == "cos") x_out_kernel_fixed[i] = activation_cos(x_in_fixed[i]);
            if (activation_name == "identity") x_out_kernel_fixed[i] = activation_identity(x_in_fixed[i]);
        }

        float x_out_kernel_float[SIZE];
        cast_1d<SIZE>(x_out_kernel_fixed, x_out_kernel_float);

        bool pass_idividual = true;
        for (int i = 0; i < SIZE; i++) {
            float error = std::fabs(x_out_float[i] - x_out_kernel_float[i]);
            if (error > EPS) {
                printf("Activation %s failed at index %d: %f != %f\n", activation_name.c_str(), i, x_out_float[i], x_out_kernel_float[i]);
                pass_idividual &= false;
            }
        }
        if (!pass_idividual) {
            printf("Activation %s failed\n", activation_name.c_str());
        }

        pass_activations.push_back(pass_idividual);
        // cout << pass_activations[i] << endl;
    }

    bool pass = true;
    for (int i = 0; i < pass_activations.size(); i++) {
        pass &= pass_activations[i];
    }

    return pass;
}

bool test_mean_incremental() {
    const int MEAN_SIZE = 10;
    float mean_in_float[MEAN_SIZE];
    F_TYPE mean_in_fixed[MEAN_SIZE];
    for (int i = 0; i < MEAN_SIZE; i++) {
        mean_in_float[i] = i - MEAN_SIZE / 2;
    }
    for (int i = 0; i < MEAN_SIZE; i++) {
        mean_in_fixed[i] = F_TYPE(mean_in_float[i]);
    }

    float mean_out_gold_float;
    F_TYPE mean_out_gold_fixed;

    mean_out_gold_float = 0;
    for (int i = 0; i < MEAN_SIZE; i++) {
        mean_out_gold_float += mean_in_float[i];
    }
    mean_out_gold_float /= MEAN_SIZE;

    float mean_out_kernel_float;
    F_TYPE mean_out_kernel_fixed;

    mean_incremental_data<F_TYPE> mean_data;
    for (int i = 0; i < MEAN_SIZE; i++) {
        mean_incremental_update<F_TYPE>(mean_data, mean_in_fixed[i]);
    }
    mean_incremental_finalize<F_TYPE>(mean_data);

    mean_out_kernel_fixed = mean_data.mean;
    mean_out_kernel_float = float(mean_out_kernel_fixed);

    // check gold and kernel output are the same
    bool mean_pass = true;
    float eps = 1e-3;
    if (fabs(mean_out_gold_float - mean_out_kernel_float) > eps) {
        printf("%f != %f\n", mean_out_gold_float, mean_out_kernel_float);
        mean_pass = false;
    }
    return mean_pass;
}

bool test_variance_incremental() {
    const int VARIANCE_SIZE = 10;
    float variance_in_float[VARIANCE_SIZE];
    F_TYPE variance_in_fixed[VARIANCE_SIZE];
    for (int i = 0; i < VARIANCE_SIZE; i++) {
        variance_in_float[i] = i - VARIANCE_SIZE / 2;
    }
    for (int i = 0; i < VARIANCE_SIZE; i++) {
        variance_in_fixed[i] = F_TYPE(variance_in_float[i]);
    }

    float variance_out_gold_float;
    F_TYPE variance_out_gold_fixed;
    float mean_sum = 0;
    for (int i = 0; i < VARIANCE_SIZE; i++) {
        mean_sum += variance_in_float[i];
    }
    float mean = mean_sum / VARIANCE_SIZE;
    float mean_diff;
    float var_sum = 0;
    for (int i = 0; i < VARIANCE_SIZE; i++) {
        mean_diff = variance_in_float[i] - mean;
        var_sum += mean_diff * mean_diff;
    }
    variance_out_gold_float = var_sum / VARIANCE_SIZE;
    variance_out_gold_fixed = F_TYPE(variance_out_gold_float);

    float variance_out_kernel_float;
    F_TYPE variance_out_kernel_fixed;

    variance_incremental_data<F_TYPE> variance_data;
    for (int i = 0; i < VARIANCE_SIZE; i++) {
        variance_incremental_update<F_TYPE>(variance_data, variance_in_fixed[i]);
    }
    variance_incremental_finalize<F_TYPE>(variance_data);

    variance_out_kernel_fixed = variance_data.var;
    variance_out_kernel_float = float(variance_out_kernel_fixed);

    // check gold and kernel output are the same
    bool variance_pass = true;
    float eps = 1e-3;
    if (fabs(variance_out_gold_float - variance_out_kernel_float) > eps) {
        printf("%f != %f\n", variance_out_gold_float, variance_out_kernel_float);
        variance_pass = false;
    }
    return variance_pass;
}

bool test_sum_incremental() {
    const int SUM_SIZE = 10;
    float sum_in_float[SUM_SIZE];
    F_TYPE sum_in_fixed[SUM_SIZE];
    for (int i = 0; i < SUM_SIZE; i++) {
        sum_in_float[i] = i - SUM_SIZE / 2;
    }
    for (int i = 0; i < SUM_SIZE; i++) {
        sum_in_fixed[i] = F_TYPE(sum_in_float[i]);
    }

    float sum_out_gold_float;
    F_TYPE sum_out_gold_fixed;
    for (int i = 0; i < SUM_SIZE; i++) {
        sum_out_gold_float += sum_in_float[i];
    }
    sum_out_gold_fixed = F_TYPE(sum_out_gold_float);

    float sum_out_kernel_float;
    F_TYPE sum_out_kernel_fixed;

    sum_incremental_data<F_TYPE> sum_data;
    for (int i = 0; i < SUM_SIZE; i++) {
        sum_incremental_update<F_TYPE>(sum_data, sum_in_fixed[i]);
    }
    sum_incremental_finalize<F_TYPE>(sum_data);

    sum_out_kernel_fixed = sum_data.sum;
    sum_out_kernel_float = float(sum_out_kernel_fixed);

    // check gold and kernel output are the same
    bool sum_pass = true;
    float eps = 1e-3;
    if (fabs(sum_out_gold_float - sum_out_kernel_float) > eps) {
        printf("%f != %f\n", sum_out_gold_float, sum_out_kernel_float);
        sum_pass = false;
    }
    return sum_pass;
}

bool test_max_incremental() {
    const int MAX_SIZE = 10;
    float max_in_float[MAX_SIZE];
    F_TYPE max_in_fixed[MAX_SIZE];
    for (int i = 0; i < MAX_SIZE; i++) {
        max_in_float[i] = i - MAX_SIZE / 2;
    }
    for (int i = 0; i < MAX_SIZE; i++) {
        max_in_fixed[i] = F_TYPE(max_in_float[i]);
    }

    float max_out_gold_float;
    F_TYPE max_out_gold_fixed;

    max_out_gold_float = max_in_float[0];
    for (int i = 1; i < MAX_SIZE; i++) {
        if (max_in_float[i] > max_out_gold_float) {
            max_out_gold_float = max_in_float[i];
        }
    }
    max_out_gold_fixed = F_TYPE(max_out_gold_float);

    float max_out_kernel_float;
    F_TYPE max_out_kernel_fixed;

    max_incremental_data<F_TYPE> max_data;
    for (int i = 0; i < MAX_SIZE; i++) {
        max_incremental_update<F_TYPE>(max_data, max_in_fixed[i]);
    }
    max_incremental_finalize<F_TYPE>(max_data);

    max_out_kernel_fixed = max_data.max;
    max_out_kernel_float = float(max_out_kernel_fixed);

    // check gold and kernel output are the same
    bool max_pass = true;
    float eps = 1e-3;
    if (fabs(max_out_gold_float - max_out_kernel_float) > eps) {
        printf("%f != %f\n", max_out_gold_float, max_out_kernel_float);
        max_pass = false;
    }
    return max_pass;
}

bool test_min_incremental() {
    const int MIN_SIZE = 10;
    float min_in_float[MIN_SIZE];
    F_TYPE min_in_fixed[MIN_SIZE];
    for (int i = 0; i < MIN_SIZE; i++) {
        min_in_float[i] = i - MIN_SIZE / 2;
    }
    for (int i = 0; i < MIN_SIZE; i++) {
        min_in_fixed[i] = F_TYPE(min_in_float[i]);
    }

    float min_out_gold_float;
    F_TYPE min_out_gold_fixed;

    min_out_gold_float = min_in_float[0];
    for (int i = 1; i < MIN_SIZE; i++) {
        if (min_in_float[i] < min_out_gold_float) {
            min_out_gold_float = min_in_float[i];
        }
    }
    min_out_gold_fixed = F_TYPE(min_out_gold_float);

    float min_out_kernel_float;
    F_TYPE min_out_kernel_fixed;

    min_incremental_data<F_TYPE> min_data;
    for (int i = 0; i < MIN_SIZE; i++) {
        min_incremental_update<F_TYPE>(min_data, min_in_fixed[i]);
    }
    min_incremental_finalize<F_TYPE>(min_data);

    min_out_kernel_fixed = min_data.min;
    min_out_kernel_float = float(min_out_kernel_fixed);

    // check gold and kernel output are the same
    bool min_pass = true;
    float eps = 1e-3;
    if (fabs(min_out_gold_float - min_out_kernel_float) > eps) {
        printf("%f != %f\n", min_out_gold_float, min_out_kernel_float);
        min_pass = false;
    }
    return min_pass;
}

bool test_linear() {
    const int LINEAR_SIZE_IN = 10;
    const int LINEAR_SIZE_OUT = 20;

    float linear_in_float[LINEAR_SIZE_IN];
    F_TYPE linear_in_fixed[LINEAR_SIZE_IN];
    for (int i = 0; i < LINEAR_SIZE_IN; i++) {
        linear_in_float[i] = i - LINEAR_SIZE_IN / 2;
    }
    for (int i = 0; i < LINEAR_SIZE_IN; i++) {
        linear_in_fixed[i] = F_TYPE(linear_in_float[i]);
    }

    float linear_weight_float[LINEAR_SIZE_OUT][LINEAR_SIZE_IN];
    F_TYPE linear_weight_fixed[LINEAR_SIZE_OUT][LINEAR_SIZE_IN];
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        for (int j = 0; j < LINEAR_SIZE_IN; j++) {
            linear_weight_float[i][j] = i * j;
        }
    }
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        for (int j = 0; j < LINEAR_SIZE_IN; j++) {
            linear_weight_fixed[i][j] = F_TYPE(linear_weight_float[i][j]);
        }
    }

    float linear_biased_float[LINEAR_SIZE_OUT];
    F_TYPE linear_biased_fixed[LINEAR_SIZE_OUT];
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_biased_float[i] = i;
    }
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_biased_fixed[i] = F_TYPE(linear_biased_float[i]);
    }

    float linear_out_gold_float[LINEAR_SIZE_OUT];
    F_TYPE linear_out_gold_fixed[LINEAR_SIZE_OUT];
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_out_gold_float[i] = 0;
        for (int j = 0; j < LINEAR_SIZE_IN; j++) {
            linear_out_gold_float[i] += linear_in_float[j] * linear_weight_float[i][j];
        }
        linear_out_gold_float[i] += linear_biased_float[i];
    }
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_out_gold_fixed[i] = F_TYPE(linear_out_gold_float[i]);
    }

    float linear_out_kernel_float[LINEAR_SIZE_OUT];
    F_TYPE linear_out_kernel_fixed[LINEAR_SIZE_OUT];

    linear<LINEAR_SIZE_IN, LINEAR_SIZE_OUT, 2, 5>(linear_in_fixed, linear_out_kernel_fixed, linear_weight_fixed, linear_biased_fixed);

    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_out_kernel_float[i] = float(linear_out_kernel_fixed[i]);
    }

    // check gold and kernel output are the same
    bool linear_pass = true;
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        if (linear_out_gold_float[i] != linear_out_kernel_float[i]) {
            printf("%f != %f\n", linear_out_gold_float[i], linear_out_kernel_float[i]);
            linear_pass = false;
        }
    }
    return linear_pass;
}

bool test_linear_buffered(){
    const int LINEAR_SIZE_IN = 10;
    const int LINEAR_SIZE_OUT = 20;

    float linear_in_float[LINEAR_SIZE_IN];
    F_TYPE linear_in_fixed[LINEAR_SIZE_IN];
    for (int i = 0; i < LINEAR_SIZE_IN; i++) {
        linear_in_float[i] = i - LINEAR_SIZE_IN / 2;
    }
    for (int i = 0; i < LINEAR_SIZE_IN; i++) {
        linear_in_fixed[i] = F_TYPE(linear_in_float[i]);
    }

    float linear_weight_float[LINEAR_SIZE_OUT][LINEAR_SIZE_IN];
    F_TYPE linear_weight_fixed[LINEAR_SIZE_OUT][LINEAR_SIZE_IN];
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        for (int j = 0; j < LINEAR_SIZE_IN; j++) {
            linear_weight_float[i][j] = i * j;
        }
    }
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        for (int j = 0; j < LINEAR_SIZE_IN; j++) {
            linear_weight_fixed[i][j] = F_TYPE(linear_weight_float[i][j]);
        }
    }

    float linear_biased_float[LINEAR_SIZE_OUT];
    F_TYPE linear_biased_fixed[LINEAR_SIZE_OUT];
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_biased_float[i] = i;
    }
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_biased_fixed[i] = F_TYPE(linear_biased_float[i]);
    }

    float linear_out_gold_float[LINEAR_SIZE_OUT];
    F_TYPE linear_out_gold_fixed[LINEAR_SIZE_OUT];
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_out_gold_float[i] = 0;
        for (int j = 0; j < LINEAR_SIZE_IN; j++) {
            linear_out_gold_float[i] += linear_in_float[j] * linear_weight_float[i][j];
        }
        linear_out_gold_float[i] += linear_biased_float[i];
    }
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_out_gold_fixed[i] = F_TYPE(linear_out_gold_float[i]);
    }

    float linear_out_kernel_float[LINEAR_SIZE_OUT];
    F_TYPE linear_out_kernel_fixed[LINEAR_SIZE_OUT];

    linear_buffered<LINEAR_SIZE_IN, LINEAR_SIZE_OUT, 2, 5>(linear_in_fixed, linear_out_kernel_fixed, linear_weight_fixed, linear_biased_fixed);

    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        linear_out_kernel_float[i] = float(linear_out_kernel_fixed[i]);
    }

    // check gold and kernel output are the same
    bool pass = true;
    for (int i = 0; i < LINEAR_SIZE_OUT; i++) {
        if (linear_out_gold_float[i] != linear_out_kernel_float[i]) {
            printf("%f != %f\n", linear_out_gold_float[i], linear_out_kernel_float[i]);
            pass = false;
        }
    }
    return pass;
}

bool test_compute_degree_tables() {

    const int max_nodes = 1000;
    const int max_edges = 1000;

    std::ifstream f_tb_max_nodes("./tb_data/tb_max_nodes.txt");
    std::ifstream f_tb_max_edges("./tb_data/tb_max_edges.txt");
    std::ifstream f_tb_num_nodes("./tb_data/tb_num_nodes.txt");
    std::ifstream f_tb_num_edges("./tb_data/tb_num_edges.txt");
    std::ifstream f_tb_coo_matrix("./tb_data/tb_coo_matrix.txt");
    std::ifstream f_tb_in_degree_table("./tb_data/tb_in_degree_table.txt");
    std::ifstream f_tb_out_degree_table("./tb_data/tb_out_degree_table.txt");

    int num_nodes;
    f_tb_num_nodes >> num_nodes;
    int num_edges;
    f_tb_num_edges >> num_edges;

    // printf("max_nodes: %d\n", max_nodes);
    // printf("max_edges: %d\n", max_edges);
    // printf("num_nodes: %d\n", num_nodes);
    // printf("num_edges: %d\n", num_edges);

    int coo_matrix[max_edges][2] = {0};
    load_data_var_2d<max_edges, 2>("./tb_data/tb_coo_matrix.bin", coo_matrix, num_edges, 2);

    int in_degree_table_gold[max_nodes] = {0};
    int out_degree_table_gold[max_nodes] = {0};

    load_data_var_1d<max_nodes>("./tb_data/tb_in_degree_table.bin", in_degree_table_gold, num_nodes);
    load_data_var_1d<max_nodes>("./tb_data/tb_out_degree_table.bin", out_degree_table_gold, num_nodes);

    f_tb_max_nodes.close();
    f_tb_max_edges.close();
    f_tb_num_nodes.close();
    f_tb_num_edges.close();
    f_tb_coo_matrix.close();
    f_tb_in_degree_table.close();
    f_tb_out_degree_table.close();

    int in_degree_table_kernel[max_nodes] = {0};
    int out_degree_table_kernel[max_nodes] = {0};
    compute_degree_tables<max_nodes, max_edges>(coo_matrix, in_degree_table_kernel, out_degree_table_kernel, num_nodes, num_edges);

    bool degree_tables_pass = true;
    for (int i = 0; i < num_nodes; i++) {
        if (in_degree_table_gold[i] != in_degree_table_kernel[i]) {
            printf("in_degree_table_gold[%d]: %d != in_degree_table_kernel[%d]: %d\n", i, in_degree_table_gold[i], i, in_degree_table_kernel[i]);
            degree_tables_pass = false;
        }
        if (out_degree_table_gold[i] != out_degree_table_kernel[i]) {
            printf("out_degree_table_gold[%d]: %d != out_degree_table_kernel[%d]: %d\n", i, out_degree_table_gold[i], i, out_degree_table_kernel[i]);
            degree_tables_pass = false;
        }
    }

    return degree_tables_pass;
}

bool test_compute_neighbor_tables() {

    const int max_nodes = 1000;
    const int max_edges = 1000;

    std::ifstream f_tb_max_nodes("./tb_data/tb_max_nodes.txt");
    std::ifstream f_tb_max_edges("./tb_data/tb_max_edges.txt");
    std::ifstream f_tb_num_nodes("./tb_data/tb_num_nodes.txt");
    std::ifstream f_tb_num_edges("./tb_data/tb_num_edges.txt");

    int num_nodes;
    f_tb_num_nodes >> num_nodes;
    int num_edges;
    f_tb_num_edges >> num_edges;

    int coo_matrix[max_edges][2] = {0};
    load_data_var_2d<max_edges, 2>("./tb_data/tb_coo_matrix.bin", coo_matrix, num_edges, 2);

    f_tb_max_nodes.close();
    f_tb_max_edges.close();
    f_tb_num_nodes.close();
    f_tb_num_edges.close();

    int in_degree_table_kernel[max_nodes] = {0};
    int out_degree_table_kernel[max_nodes] = {0};
    compute_degree_tables<
        max_nodes,
        max_edges>(
        coo_matrix,
        in_degree_table_kernel,
        out_degree_table_kernel,
        num_nodes, num_edges);

    int neighbor_table_offsets_kernel[max_nodes] = {0};
    int neighbor_table_kernel[max_edges] = {0};

    compute_neighbor_tables<
        max_nodes,
        max_edges>(
        coo_matrix,
        in_degree_table_kernel,
        out_degree_table_kernel,
        neighbor_table_offsets_kernel,
        neighbor_table_kernel,
        num_nodes,
        num_edges);

    int neighbor_table_offsets_gold[max_nodes] = {0};
    int neighbor_table_gold[max_edges] = {0};

    load_data_var_1d<max_nodes>("./tb_data/tb_neighbor_table_offsets.bin", neighbor_table_offsets_gold, num_nodes);
    load_data_var_1d<max_edges>("./tb_data/tb_neighbor_table.bin", neighbor_table_gold, num_edges);

    bool neighbor_tables_pass = true;
    for (int i = 0; i < num_nodes; i++) {
        if (neighbor_table_offsets_gold[i] != neighbor_table_offsets_kernel[i]) {
            printf("neighbor_table_offsets_gold[%d]: %d != neighbor_table_offsets_kernel[%d]: %d\n", i, neighbor_table_offsets_gold[i], i, neighbor_table_offsets_kernel[i]);
            neighbor_tables_pass = false;
        }
        if (neighbor_table_gold[i] != neighbor_table_kernel[i]) {
            printf("neighbor_table_gold[%d]: %d != neighbor_table_kernel[%d]: %d\n", i, neighbor_table_gold[i], i, neighbor_table_kernel[i]);
            neighbor_tables_pass = false;
        }
    }

    return neighbor_tables_pass;
}

bool test_gcn_conv() {

    const int max_nodes = 1000;
    const int max_edges = 1000;

    std::ifstream f_tb_max_nodes("./tb_data/tb_max_nodes.txt");
    std::ifstream f_tb_max_edges("./tb_data/tb_max_edges.txt");
    std::ifstream f_tb_num_nodes("./tb_data/tb_num_nodes.txt");
    std::ifstream f_tb_num_edges("./tb_data/tb_num_edges.txt");
    std::ifstream f_tb_input_node_feature_size("./tb_data/tb_input_node_feature_size.txt");
    std::ifstream f_tb_output_feature_size("./tb_data/tb_output_feature_size.txt");


    int num_nodes;
    f_tb_num_nodes >> num_nodes;
    int num_edges;
    f_tb_num_edges >> num_edges;

    int coo_matrix[max_edges][2];
    load_data_var_2d<max_edges, 2>("./tb_data/tb_coo_matrix.bin", coo_matrix, num_edges, 2);

    int in_degree_table[max_nodes];
    int out_degree_table[max_nodes];
    load_data_var_1d<max_nodes>("./tb_data/tb_in_degree_table.bin", in_degree_table, num_nodes);
    load_data_var_1d<max_nodes>("./tb_data/tb_out_degree_table.bin", out_degree_table, num_nodes);

    int neighbor_table_offsets[max_nodes];
    int neighbor_table[max_edges];
    load_data_var_1d<max_nodes>("./tb_data/tb_neighbor_table_offsets.bin", neighbor_table_offsets, num_nodes);
    load_data_var_1d<max_edges>("./tb_data/tb_neighbor_table.bin", neighbor_table, num_edges);

    int input_node_feature_size;
    f_tb_input_node_feature_size >> input_node_feature_size;
    const int input_node_feature_size_const = 8;
    // assert((input_node_feature_size == input_node_feature_size_const), "input_node_feature_size != input_node_feature_size_const");

    float input_node_features[max_nodes][input_node_feature_size_const];
    load_data_2d<max_nodes, input_node_feature_size_const>("./tb_data/tb_input_node_features.bin", input_node_features);
    F_TYPE input_node_features_fixed[max_nodes][input_node_feature_size_const];
    cast_2d<max_nodes, input_node_feature_size_const, float, F_TYPE>(input_node_features, input_node_features_fixed);

    int output_feature_size;
    f_tb_output_feature_size >> output_feature_size;
    const int output_feature_size_const = 16;
    // assert((output_feature_size == output_feature_size_const), "output_feature_size != output_feature_size_const");

    float gcn_weights[output_feature_size_const][input_node_feature_size_const];
    load_data_2d<output_feature_size_const, input_node_feature_size_const>("./tb_data/tb_gcn_weights.bin", gcn_weights);
    F_TYPE gcn_weights_fixed[output_feature_size_const][input_node_feature_size_const];
    cast_2d<output_feature_size_const, input_node_feature_size_const, float, F_TYPE>(gcn_weights, gcn_weights_fixed);

    float gcn_bias[output_feature_size_const];
    load_data_1d<output_feature_size_const>("./tb_data/tb_gcn_bias.bin", gcn_bias);
    F_TYPE gcn_bias_fixed[output_feature_size_const];
    cast_1d<output_feature_size_const, float, F_TYPE>(gcn_bias, gcn_bias_fixed);

    float gcn_output_gold[max_nodes][output_feature_size_const];
    load_data_2d<max_nodes, output_feature_size_const>("./tb_data/tb_gcn_output.bin", gcn_output_gold);

    // close files
    f_tb_max_nodes.close();
    f_tb_max_edges.close();
    f_tb_num_nodes.close();
    f_tb_num_edges.close();
    f_tb_input_node_feature_size.close();
    f_tb_output_feature_size.close();

    F_TYPE gcn_output_kernel_fixed[max_nodes][output_feature_size_const];
    gcn_conv<max_nodes, max_edges, input_node_feature_size_const, output_feature_size_const, F_TYPE>(
        num_nodes,
        num_edges,
        input_node_features_fixed,
        gcn_output_kernel_fixed,
        coo_matrix,
        neighbor_table_offsets,
        neighbor_table,
        in_degree_table,
        out_degree_table,
        gcn_weights_fixed,
        gcn_bias_fixed);

    float gcn_output_kernel[max_nodes][output_feature_size_const];
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_feature_size_const; j++) {
            gcn_output_kernel[i][j] = float(gcn_output_kernel_fixed[i][j]);
        }
    }

    bool gcn_pass = true;
    float eps = 1e-3;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_feature_size_const; j++) {
            if (fabs(gcn_output_gold[i][j] - gcn_output_kernel[i][j]) > eps) {
                gcn_pass = false;
            }
        }
    }

    return gcn_pass;
}

bool test_gin_conv() {

    const int max_nodes = 1000;
    const int max_edges = 1000;

    // open files
    std::ifstream f_tb_max_nodes("./tb_data/tb_max_nodes.txt");
    std::ifstream f_tb_max_edges("./tb_data/tb_max_edges.txt");
    std::ifstream f_tb_num_nodes("./tb_data/tb_num_nodes.txt");
    std::ifstream f_tb_num_edges("./tb_data/tb_num_edges.txt");
    std::ifstream f_tb_input_node_feature_size("./tb_data/tb_input_node_feature_size.txt");
    std::ifstream f_tb_output_feature_size("./tb_data/tb_output_feature_size.txt");
    std::ifstream f_tb_gin_hidden_feature_size("./tb_data/tb_gin_hidden_feature_size.txt");

    int num_nodes;
    f_tb_num_nodes >> num_nodes;
    int num_edges;
    f_tb_num_edges >> num_edges;

    int coo_matrix[max_edges][2];
    load_data_2d<max_edges, 2>("./tb_data/tb_coo_matrix.bin", coo_matrix);

    int in_degree_table[max_nodes];
    int out_degree_table[max_nodes];
    load_data_1d<max_nodes>("./tb_data/tb_in_degree_table.bin", in_degree_table);
    load_data_1d<max_nodes>("./tb_data/tb_out_degree_table.bin", out_degree_table);

    int neighbor_table_offsets[max_nodes];
    int neighbor_table[max_edges];
    load_data_var_1d<max_nodes>("./tb_data/tb_neighbor_table_offsets.bin", neighbor_table_offsets, num_nodes);
    load_data_var_1d<max_edges>("./tb_data/tb_neighbor_table.bin", neighbor_table, num_edges);

    int input_node_feature_size;
    f_tb_input_node_feature_size >> input_node_feature_size;
    const int input_node_feature_size_const = 8;

    float input_node_features[max_nodes][input_node_feature_size_const];
    load_data_2d<max_nodes, input_node_feature_size_const>("./tb_data/tb_input_node_features.bin", input_node_features);
    F_TYPE input_node_features_fixed[max_nodes][input_node_feature_size_const];
    cast_2d<max_nodes, input_node_feature_size_const, float, F_TYPE>(input_node_features, input_node_features_fixed);

    int output_feature_size;
    f_tb_output_feature_size >> output_feature_size;
    const int output_feature_size_const = 16;

    int gin_hidden_feature_size;
    f_tb_gin_hidden_feature_size >> gin_hidden_feature_size;
    const int gin_hidden_feature_size_const = 64;

    float gin_mlp_0_weights[gin_hidden_feature_size_const][input_node_feature_size_const];
    load_data_2d<gin_hidden_feature_size_const, input_node_feature_size_const>("./tb_data/tb_gin_mlp_0_weights.bin", gin_mlp_0_weights);
    F_TYPE gin_mlp_0_weights_fixed[gin_hidden_feature_size_const][input_node_feature_size_const];
    cast_2d<gin_hidden_feature_size_const, input_node_feature_size_const, float, F_TYPE>(gin_mlp_0_weights, gin_mlp_0_weights_fixed);

    float gin_mlp_0_bias[gin_hidden_feature_size_const];
    load_data_1d<gin_hidden_feature_size_const>("./tb_data/tb_gin_mlp_0_bias.bin", gin_mlp_0_bias);
    F_TYPE gin_mlp_0_bias_fixed[gin_hidden_feature_size_const];
    cast_1d<gin_hidden_feature_size_const, float, F_TYPE>(gin_mlp_0_bias, gin_mlp_0_bias_fixed);

    float gin_mlp_1_weights[output_feature_size_const][gin_hidden_feature_size_const];
    load_data_2d<output_feature_size_const, gin_hidden_feature_size_const>("./tb_data/tb_gin_mlp_1_weights.bin", gin_mlp_1_weights);
    F_TYPE gin_mlp_1_weights_fixed[output_feature_size_const][gin_hidden_feature_size_const];
    cast_2d<output_feature_size_const, gin_hidden_feature_size_const, float, F_TYPE>(gin_mlp_1_weights, gin_mlp_1_weights_fixed);

    float gin_mlp_1_bias[output_feature_size_const];
    load_data_1d<output_feature_size_const>("./tb_data/tb_gin_mlp_1_bias.bin", gin_mlp_1_bias);
    F_TYPE gin_mlp_1_bias_fixed[output_feature_size_const];
    cast_1d<output_feature_size_const, float, F_TYPE>(gin_mlp_1_bias, gin_mlp_1_bias_fixed);

    float gin_eps_array[1];
    float gin_eps;
    F_TYPE gin_eps_fixed;
    // f_tb_gin_eps >> gin_eps;
    // F_TYPE gin_eps_fixed = F_TYPE(gin_eps);
    load_data_1d<1>("./tb_data/tb_gin_eps.bin", gin_eps_array);
    gin_eps = gin_eps_array[0];
    gin_eps_fixed = F_TYPE(gin_eps);

    float gin_output_gold[max_nodes][output_feature_size_const];
    load_data_2d<max_nodes, output_feature_size_const>("./tb_data/tb_gin_output.bin", gin_output_gold);

    // close files
    f_tb_max_nodes.close();
    f_tb_max_edges.close();
    f_tb_num_nodes.close();
    f_tb_num_edges.close();
    f_tb_input_node_feature_size.close();
    f_tb_output_feature_size.close();
    f_tb_gin_hidden_feature_size.close();

    F_TYPE gin_output_kernel_fixed[max_nodes][output_feature_size_const];
    gin_conv<max_nodes, max_edges, input_node_feature_size_const, output_feature_size_const, gin_hidden_feature_size_const>(
        num_nodes,
        num_edges,
        input_node_features_fixed,
        gin_output_kernel_fixed,
        coo_matrix,
        neighbor_table_offsets,
        neighbor_table,
        in_degree_table,
        out_degree_table,
        gin_mlp_0_weights_fixed,
        gin_mlp_0_bias_fixed,
        gin_mlp_1_weights_fixed,
        gin_mlp_1_bias_fixed,
        gin_eps_fixed);

    float gin_output_kernel[max_nodes][output_feature_size_const];
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_feature_size_const; j++) {
            gin_output_kernel[i][j] = float(gin_output_kernel_fixed[i][j]);
        }
    }

    bool gin_pass = true;
    float eps = 1e-3;
    float error = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_feature_size_const; j++) {
            error += fabs(gin_output_kernel[i][j] - gin_output_gold[i][j]);
            if (fabs(gin_output_kernel[i][j] - gin_output_gold[i][j]) > eps) {
                gin_pass = false;
            }
        }
    }
    error /= (num_nodes * output_feature_size_const);

    return gin_pass;
}

bool test_pna_conv() {
    const int max_nodes = 1000;
    const int max_edges = 1000;

    // open files
    std::ifstream f_tb_max_nodes("./tb_data/tb_max_nodes.txt");
    std::ifstream f_tb_max_edges("./tb_data/tb_max_edges.txt");
    std::ifstream f_tb_num_nodes("./tb_data/tb_num_nodes.txt");
    std::ifstream f_tb_num_edges("./tb_data/tb_num_edges.txt");
    std::ifstream f_tb_coo_matrix("./tb_data/tb_coo_matrix.txt");
    std::ifstream f_tb_in_degree_table("./tb_data/tb_in_degree_table.txt");
    std::ifstream f_tb_out_degree_table("./tb_data/tb_out_degree_table.txt");
    std::ifstream f_tb_input_node_feature_size("./tb_data/tb_input_node_feature_size.txt");
    std::ifstream f_tb_input_node_features("./tb_data/tb_input_node_features.txt");
    std::ifstream f_tb_output_feature_size("./tb_data/tb_output_feature_size.txt");

    int num_nodes;
    f_tb_num_nodes >> num_nodes;
    int num_edges;
    f_tb_num_edges >> num_edges;

    int coo_matrix[max_edges][2];
    load_data_2d<max_edges, 2>("./tb_data/tb_coo_matrix.bin", coo_matrix);

    int in_degree_table[max_nodes];
    int out_degree_table[max_nodes];
    load_data_1d<max_nodes>("./tb_data/tb_in_degree_table.bin", in_degree_table);
    load_data_1d<max_nodes>("./tb_data/tb_out_degree_table.bin", out_degree_table);

    int neighbor_table_offsets[max_nodes];
    int neighbor_table[max_edges];
    load_data_var_1d<max_nodes>("./tb_data/tb_neighbor_table_offsets.bin", neighbor_table_offsets, num_nodes);
    load_data_var_1d<max_edges>("./tb_data/tb_neighbor_table.bin", neighbor_table, num_edges);

    int input_node_feature_size;
    f_tb_input_node_feature_size >> input_node_feature_size;
    const int input_node_feature_size_const = 8;

    float input_node_features[max_nodes][input_node_feature_size_const];
    load_data_2d<max_nodes, input_node_feature_size_const>("./tb_data/tb_input_node_features.bin", input_node_features);
    F_TYPE input_node_features_fixed[max_nodes][input_node_feature_size_const];
    cast_2d<max_nodes, input_node_feature_size_const, float, F_TYPE>(input_node_features, input_node_features_fixed);

    int output_feature_size;
    f_tb_output_feature_size >> output_feature_size;
    const int output_feature_size_const = 16;

    const int pna_transform_in_size = input_node_feature_size_const * 2;
    const int pna_transform_out_size = input_node_feature_size_const;


    float pna_transform_lin_weights[pna_transform_out_size][pna_transform_in_size];
    load_data_2d<pna_transform_out_size, pna_transform_in_size>("./tb_data/tb_pna_transform_lin_weights.bin", pna_transform_lin_weights);
    F_TYPE pna_transform_lin_weights_fixed[pna_transform_out_size][pna_transform_in_size];
    cast_2d<pna_transform_out_size, pna_transform_in_size, float, F_TYPE>(pna_transform_lin_weights, pna_transform_lin_weights_fixed);



    float pna_transform_lin_bias[pna_transform_out_size];
    load_data_1d<pna_transform_out_size>("./tb_data/tb_pna_transform_lin_bias.bin", pna_transform_lin_bias);
    F_TYPE pna_transform_lin_bias_fixed[pna_transform_out_size];
    cast_1d<pna_transform_out_size, float, F_TYPE>(pna_transform_lin_bias, pna_transform_lin_bias_fixed);


    const int pna_apply_in_size = (input_node_feature_size_const * 4 * 3) + input_node_feature_size_const;
    const int pna_apply_out_size = output_feature_size_const;


    float pna_apply_lin_weights[pna_apply_out_size][pna_apply_in_size];
    load_data_2d<pna_apply_out_size, pna_apply_in_size>("./tb_data/tb_pna_apply_lin_weights.bin", pna_apply_lin_weights);
    F_TYPE pna_apply_lin_weights_fixed[pna_apply_out_size][pna_apply_in_size];
    cast_2d<pna_apply_out_size, pna_apply_in_size, float, F_TYPE>(pna_apply_lin_weights, pna_apply_lin_weights_fixed);


    float pna_apply_lin_bias[pna_apply_out_size];
    load_data_1d<pna_apply_out_size>("./tb_data/tb_pna_apply_lin_bias.bin", pna_apply_lin_bias);
    F_TYPE pna_apply_lin_bias_fixed[pna_apply_out_size];
    cast_1d<pna_apply_out_size, float, F_TYPE>(pna_apply_lin_bias, pna_apply_lin_bias_fixed);


    float pna_final_lin_weights[output_feature_size_const][pna_apply_out_size];
    load_data_2d<output_feature_size_const, pna_apply_out_size>("./tb_data/tb_pna_final_lin_weights.bin", pna_final_lin_weights);
    F_TYPE pna_final_lin_weights_fixed[output_feature_size_const][pna_apply_out_size];
    cast_2d<output_feature_size_const, pna_apply_out_size, float, F_TYPE>(pna_final_lin_weights, pna_final_lin_weights_fixed);


    float pna_final_lin_bias[output_feature_size_const];
    load_data_1d<output_feature_size_const>("./tb_data/tb_pna_final_lin_bias.bin", pna_final_lin_bias);
    F_TYPE pna_final_lin_bias_fixed[output_feature_size_const];
    cast_1d<output_feature_size_const, float, F_TYPE>(pna_final_lin_bias, pna_final_lin_bias_fixed);


    // float pna_avg_degree_log;
    // f_tb_pna_avg_degree_log >> pna_avg_degree_log;
    // F_TYPE pna_avg_degree_log_fixed = F_TYPE(pna_avg_degree_log);
    float pna_avg_degree_log_array[1];
    float pna_avg_degree_log;
    F_TYPE pna_avg_degree_log_fixed;
    load_data_1d<1>("./tb_data/tb_pna_avg_degree_log.bin", pna_avg_degree_log_array);
    pna_avg_degree_log = pna_avg_degree_log_array[0];
    pna_avg_degree_log_fixed = F_TYPE(pna_avg_degree_log);


    float pna_output_gold[max_nodes][output_feature_size_const];
    load_data_2d<max_nodes, output_feature_size_const>("./tb_data/tb_pna_output.bin", pna_output_gold);


    F_TYPE pna_output_kernel_fixed[max_nodes][output_feature_size_const];
    pna_conv<max_nodes, max_edges, input_node_feature_size_const, output_feature_size_const, pna_transform_in_size, pna_transform_out_size, pna_apply_in_size, pna_apply_out_size>(
        num_nodes,
        num_edges,
        input_node_features_fixed,
        pna_output_kernel_fixed,
        coo_matrix,
        neighbor_table_offsets,
        neighbor_table,
        in_degree_table,
        out_degree_table,
        pna_transform_lin_weights_fixed,
        pna_transform_lin_bias_fixed,
        pna_apply_lin_weights_fixed,
        pna_apply_lin_bias_fixed,
        pna_final_lin_weights_fixed,
        pna_final_lin_bias_fixed,
        pna_avg_degree_log_fixed);

    float pna_output_kernel[max_nodes][output_feature_size_const];
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_feature_size_const; j++) {
            pna_output_kernel[i][j] = float(pna_output_kernel_fixed[i][j]);
        }
    }

    bool pna_pass = true;
    float eps = 1e-2;
    float avg_diff = 0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < output_feature_size_const; j++) {
            float diff = fabs(pna_output_kernel[i][j] - pna_output_gold[i][j]);
            avg_diff += diff;
            if (fabs(pna_output_kernel[i][j] - pna_output_gold[i][j]) > eps) {
                printf("pna_output_kernel[%d][%d] = %f, pna_output_gold[%d][%d] = %f\n", i, j, pna_output_kernel[i][j], i, j, pna_output_gold[i][j]);
                pna_pass = false;
            }
        }
    }
    avg_diff /= (num_nodes * output_feature_size_const);
    // printf("avg_diff = %f\n", avg_diff);

    return pna_pass;
}

bool test_sage_conv(){

    const int max_nodes = 1000;
    const int max_edges = 1000;

    std::ifstream f_tb_num_nodes("./tb_data/tb_num_nodes.txt");
    std::ifstream f_tb_num_edges("./tb_data/tb_num_edges.txt");

    int num_nodes;
    int num_edges;
    f_tb_num_nodes >> num_nodes;
    f_tb_num_edges >> num_edges;

    f_tb_num_nodes.close();
    f_tb_num_edges.close();

    int coo_matrix[max_edges][2];
    load_data_2d<max_edges, 2>("./tb_data/tb_coo_matrix.bin", coo_matrix);

    int in_degree_table[max_nodes];
    int out_degree_table[max_nodes];
    load_data_1d<max_nodes>("./tb_data/tb_in_degree_table.bin", in_degree_table);
    load_data_1d<max_nodes>("./tb_data/tb_out_degree_table.bin", out_degree_table);

    int neighbor_table_offsets[max_nodes];
    int neighbor_table[max_edges];
    load_data_var_1d<max_nodes>("./tb_data/tb_neighbor_table_offsets.bin", neighbor_table_offsets, num_nodes);
    load_data_var_1d<max_edges>("./tb_data/tb_neighbor_table.bin", neighbor_table, num_edges);

    const int input_node_feature_size_const = 8;

    float input_node_features[max_nodes][input_node_feature_size_const];
    load_data_2d<max_nodes, input_node_feature_size_const>("./tb_data/tb_input_node_features.bin", input_node_features);
    F_TYPE input_node_features_fixed[max_nodes][input_node_feature_size_const];
    cast_2d<max_nodes, input_node_feature_size_const, float, F_TYPE>(input_node_features, input_node_features_fixed);

    const int output_feature_size_const = 16;

    float sage_neighbor_lin_weights[output_feature_size_const][input_node_feature_size_const];
    load_data_2d<output_feature_size_const, input_node_feature_size_const>("./tb_data/tb_sage_neighbor_lin_weights.bin", sage_neighbor_lin_weights);
    F_TYPE sage_neighbor_lin_weights_fixed[output_feature_size_const][input_node_feature_size_const];
    cast_2d<output_feature_size_const, input_node_feature_size_const, float, F_TYPE>(sage_neighbor_lin_weights, sage_neighbor_lin_weights_fixed);
    // printf("sage_neighbor_lin_weights\n");
    // print_2d<output_feature_size_const, input_node_feature_size_const>(sage_neighbor_lin_weights);
    
    float sage_neighbor_lin_bias[output_feature_size_const];
    load_data_1d<output_feature_size_const>("./tb_data/tb_sage_neighbor_lin_bias.bin", sage_neighbor_lin_bias);
    F_TYPE sage_neighbor_lin_bias_fixed[output_feature_size_const];
    cast_1d<output_feature_size_const, float, F_TYPE>(sage_neighbor_lin_bias, sage_neighbor_lin_bias_fixed);
    // printf("sage_neighbor_lin_bias\n");
    // print_1d<output_feature_size_const>(sage_neighbor_lin_bias);

    float sage_self_lin_weights[output_feature_size_const][input_node_feature_size_const];
    load_data_2d<output_feature_size_const, input_node_feature_size_const>("./tb_data/tb_sage_self_lin_weights.bin", sage_self_lin_weights);
    F_TYPE sage_self_lin_weights_fixed[output_feature_size_const][input_node_feature_size_const];
    cast_2d<output_feature_size_const, input_node_feature_size_const, float, F_TYPE>(sage_self_lin_weights, sage_self_lin_weights_fixed);
    // printf("sage_self_lin_weights\n");
    // print_2d<output_feature_size_const, input_node_feature_size_const>(sage_self_lin_weights);

    // float sage_self_lin_bias[output_feature_size_const];
    // load_data_1d<output_feature_size_const>("./tb_data/tb_sage_self_lin_bias.bin", sage_self_lin_bias);
    // F_TYPE sage_self_lin_bias_fixed[output_feature_size_const];
    // cast_1d<output_feature_size_const, float, F_TYPE>(sage_self_lin_bias, sage_self_lin_bias_fixed);

    float sage_output_gold[max_nodes][output_feature_size_const] = {0};
    load_data_var_2d<max_nodes, output_feature_size_const>("./tb_data/tb_sage_output.bin", sage_output_gold, num_nodes, output_feature_size_const);
    // printf("sage_output_gold\n");
    // print_2d<max_nodes, output_feature_size_const>(sage_output_gold);

    F_TYPE sage_output_kernel_fixed[max_nodes][output_feature_size_const];
    sage_conv<
        max_nodes,
        max_edges,
        input_node_feature_size_const,
        output_feature_size_const,
        F_TYPE
    >(
        num_nodes,
        num_edges,
        input_node_features_fixed,
        sage_output_kernel_fixed,
        coo_matrix,
        neighbor_table_offsets,
        neighbor_table,
        in_degree_table,
        out_degree_table,
        sage_neighbor_lin_weights_fixed,
        sage_neighbor_lin_bias_fixed,
        sage_self_lin_weights_fixed
    );

    float sage_output_kernel[max_nodes][output_feature_size_const];
    for (int i = 0; i < num_nodes; i++){
        for (int j = 0; j < output_feature_size_const; j++){
            sage_output_kernel[i][j] = float(sage_output_kernel_fixed[i][j]);
        }
    }

    bool sage_pass = true;
    float eps = 1e-3;
    float avg_diff = 0.0;

    for(int i = 0; i < num_nodes; i++){
        for(int j = 0; j < output_feature_size_const; j++){
            avg_diff += fabs(sage_output_kernel[i][j] - sage_output_gold[i][j]);
            // printf("kernel[%d][%d] = %f, gold[%d][%d] = %f\n", i, j, sage_output_kernel[i][j], i, j, sage_output_gold[i][j]);
            if(fabs(sage_output_kernel[i][j] - sage_output_gold[i][j]) > eps){
                printf("sage_output_kernel[%d][%d] = %f, sage_output_gold[%d][%d] = %f\n", i, j, sage_output_kernel[i][j], i, j, sage_output_gold[i][j]);
                sage_pass = false;
            }
        }
    }

    avg_diff /= num_nodes*output_feature_size_const;
    // printf("avg_diff: %f\n", avg_diff);

    return sage_pass;
}

int main() {

    printf("############################\n");
    printf("### GNNBuilder Testbench ###\n");
    printf("############################\n");

    bool pass_test_activations = test_activations();
    if (pass_test_activations) {
        printf("Test Activations: PASSED\n");
    } else {
        printf("Test Activations: FAILED\n");
    }

    bool pass_test_mean_incremental = test_mean_incremental();
    if (pass_test_mean_incremental) {
        printf("Test Mean Incremental: PASSED\n");
    } else {
        printf("Test Mean Incremental: FAILED\n");
    }

    bool pass_test_variance_incremental = test_variance_incremental();
    if (pass_test_variance_incremental) {
        printf("Test Variance Incremental: PASSED\n");
    } else {
        printf("Test Variance Incremental: FAILED\n");
    }

    bool pass_test_sum_incremental = test_sum_incremental();
    if (pass_test_sum_incremental) {
        printf("Test Sum Incremental: PASSED\n");
    } else {
        printf("Test Sum Incremental: FAILED\n");
    }

    // max incremental test
    bool pass_test_max_incremental = test_max_incremental();
    if (pass_test_max_incremental) {
        printf("Test Max Incremental: PASSED\n");
    }
    else {
        printf("Test Max Incremental: FAILED\n");
    }

    bool pass_test_min_incremental = test_min_incremental();
    if (pass_test_min_incremental) {
        printf("Test Min Incremental: PASSED\n");
    }
    else {
        printf("Test Min Incremental: FAILED\n");
    }

    bool pass_test_linear = test_linear();
    if (pass_test_linear) {
        printf("pass_test_linear: PASSED\n");
    }
    else{
        printf("pass_test_linear: FAILED\n");
    }

    bool pass_test_linear_buffered = test_linear_buffered();
    if (pass_test_linear_buffered) {
        printf("pass_test_linear_buffered: PASSED\n");
    }
    else{
        printf("pass_test_linear_buffered: FAILED\n");
    }

    bool pass_test_compute_degree_tables = test_compute_degree_tables();
    if (pass_test_compute_degree_tables) {
        printf("Test Compute Degree Tables: PASSED\n");
    } else {
        printf("Test Compute Degree Tables: FAILED\n");
    }

    bool pass_test_compute_neighbor_tables = test_compute_neighbor_tables();
    if (pass_test_compute_neighbor_tables) {
        printf("Test Compute Neighbor Tables: PASSED\n");
    } else {
        printf("Test Compute Neighbor Tables: FAILED\n");
    }

    bool pass_test_gcn_conv = test_gcn_conv();
    if (pass_test_gcn_conv) {
        printf("gcn_conv: PASS\n");
    } else {
        printf("gcn_conv: FAIL\n");
    }

    bool pass_test_gin_conv = test_gin_conv();
    if (pass_test_gin_conv) {
        printf("gin_conv: PASS\n");
    } else {
        printf("gin_conv: FAIL\n");
    }

    bool pass_test_pna_conv = test_pna_conv();
    if (pass_test_pna_conv) {
        printf("pna_conv: PASS\n");
    } else {
        printf("pna_conv: FAIL\n");
    }

    bool pass_test_sage_conv = test_sage_conv();
    if (pass_test_sage_conv) {
        printf("sage_conv: PASS\n");
    } else {
        printf("sage_conv: FAIL\n");
    }

    return 0;


}