#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <istream>

namespace vrs {
    // Represents one raw record from the CSV file.
    struct Row {
        std::string company;
        std::string license;
        std::chrono::year_month_day company_inspection_expiry;
        std::string vehicle_id;
        std::chrono::year_month_day vehicle_inspection_expiry;
        std::string driver;
        std::chrono::year_month_day permit_expiry;
        std::chrono::year_month_day driver_license_expiry;
    };
    
    // Reads all rows from a CSV file at the given path.
    std::vector<Row> read_all(const std::string &path);
    // Reads a single CSV row from the input stream.
    bool read_row(std::istream& in, Row &r);
}