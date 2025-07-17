#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_94_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<94> mm_instantiator;

public:
  UnalignedM_94_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_94_MMInstantiatorWrapper__;

}  // namespace LLMMM
