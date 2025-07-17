#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_49_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<49> mm_instantiator;

public:
  UnalignedM_49_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_49_MMInstantiatorWrapper__;

}  // namespace LLMMM
