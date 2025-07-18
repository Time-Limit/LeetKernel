#include "llmmm/llmmm.cuh"

namespace LLMMM {

class AlignedMMMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<128> mm_instantiator;

public:
  AlignedMMMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __AlignedMMMInstantiatorWrapper__;

}  // namespace LLMMM
