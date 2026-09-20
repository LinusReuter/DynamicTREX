#pragma once

#include "../../../DataStructures/DynamicTimeTable/Data.h"
#include "../../DynamicTimeTable/BuildQueryData.h"
#include "../../TripBased/Query/Types.h"
#include "CellBorderIndex.h"
#include "CellStopIndex.h"
#include "CustomizationTypes.h"

namespace DynamicTB::Customization {

/**
 * @brief The two cell indices, built once and kept for the life of the process.
 *
 * Both are derived from the partition, which RT updates never change. The only per-update
 * maintenance in either is `CellBorderIndex::syncNewRoutes` -- updates mint persistent routes,
 * and a route the border index has never seen contributes no search seeds.
 */
class CellIndices {
public:
    void build(const DynamicTimeTable::Data& data) {
        borders_.build(data);
        cellStops_.build(data);
    }

    /** Pick up routes minted since the last build/sync. Call once per update, before the sweep. */
    void sync(const DynamicTimeTable::Data& data) { borders_.syncNewRoutes(data); }

    /**
     * Bundle the indices with this minute's timetable, query data and CSR. Nothing is copied;
     * the result is valid as long as all four outlive it.
     */
    CellNetwork bind(const DynamicTimeTable::Data& data, const DynamicTimeTable::Algo::DynamicQueryData& queryData,
                     TripBased::Transfers& csr) const {
        CellNetwork net;
        net.data = &data;
        net.qd = &queryData;
        net.borders = &borders_;
        net.cellStops = &cellStops_;
        net.csr = &csr;
        net.levels = data.getNumberOfLevels();
        return net;
    }

    long long byteSize() const noexcept { return borders_.byteSize() + cellStops_.byteSize(); }

private:
    CellBorderIndex borders_;
    CellStopIndex cellStops_;
};

}  // namespace DynamicTB::Customization
