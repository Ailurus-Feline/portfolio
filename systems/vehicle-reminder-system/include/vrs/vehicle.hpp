#pragma once
#include <chrono>
#include <string>

// Defines class Vehicle for storing and checking document expiry info.
namespace vrs {
    // A single vehicle and its document expiration data.
    class Vehicle {
        private:
            using date = std::chrono::year_month_day;

            static constexpr std::chrono::days reminder_time {30};

            std::string company;
            date driver_license_expiry;
            date permit_expiry;
            date inspection_expiry;

            //Returns true if dues within 30 days or has expired.
            static bool is_due_soon(const date &expiry_date) noexcept {
                using namespace std::chrono;

                const auto today = floor<days>(system_clock::now());
                const auto expiry = sys_days(expiry_date);

                return (expiry - today <= reminder_time);
            };

        public:
            explicit Vehicle(std::string name, date license, date permit, date inspection) : 
            company(name),
            driver_license_expiry(license),
            permit_expiry(permit),
            inspection_expiry(inspection)
            {};

            // Returns true if expires within 30 days.
            bool is_license_due() const noexcept {
                return is_due_soon(driver_license_expiry);
            }
            // Returns true if expires within 30 days.
            bool is_permit_due() const noexcept {
               return is_due_soon(permit_expiry);
            } 
            // Returns true if expires within 30 days.
            bool is_inspection_due() const noexcept {
               return is_due_soon(inspection_expiry);
            } 
            ~Vehicle() = default;
    };
}