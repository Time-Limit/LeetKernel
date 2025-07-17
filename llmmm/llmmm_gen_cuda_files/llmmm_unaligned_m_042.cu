#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_42_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<42> mm_instantiator;

public:
  UnalignedM_42_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_42_MMInstantiatorWrapper__;

}  // namespace LLMMM
