#include "../include/vrs/csv_reader.hpp"
#include <fstream>

namespace vrs {
    std::vector<Row> read_all(const std::string &path){
        std::vector<Row> rows;
        std::ifstream in(path);
        Row r;

        while (read_row(in, r))
            rows.push_back(r);
        
        return rows;
    }

    bool read_row(std::istream& in, Row &r) {
        
    }
}