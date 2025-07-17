#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_63_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<63> mm_instantiator;

public:
  UnalignedM_63_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_63_MMInstantiatorWrapper__;

}  // namespace LLMMM
