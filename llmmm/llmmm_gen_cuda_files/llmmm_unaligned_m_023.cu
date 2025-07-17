#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_23_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<23> mm_instantiator;

public:
  UnalignedM_23_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_23_MMInstantiatorWrapper__;

}  // namespace LLMMM
