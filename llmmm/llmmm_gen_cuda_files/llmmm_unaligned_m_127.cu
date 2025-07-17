#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_127_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<127> mm_instantiator;

public:
  UnalignedM_127_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_127_MMInstantiatorWrapper__;

}  // namespace LLMMM
